import sys
import numpy as np
import pyqtgraph as pg
import datetime
import usb.core
from PyQt6.QtCore import QSize, Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QPainterPath
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                           QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QFormLayout, 
                             QToolBar, QToolButton, QMenu, QFileDialog, QListWidget,
                             QPushButton, QListWidgetItem, QGridLayout, QCheckBox)
# --- IMPORTACIÓN DE NUESTROS MÓDULOS ---
# Hardware
from hardware.hackrf_handler import HackRFHandler
from hardware.rtlsdr_handler import RtlSdrHandler
from hardware.nuand_bladerf_handler import BladeRFHandler
from hardware.ettus_usrpb200_handler import USRPB200Handler
# DSP (Plugins)
from dsp.demoduladores.wbfm import DemoduladorWBFM
from dsp.demoduladores.wbfm_audio import DemoduladorWBFMAudio
from dsp.demoduladores.sa import SpectrumAnalyzer
from dsp.demoduladores.wifi_ag import DemoduladorWiFiAG
# Managers
from marker_manager import MarkerManager
from playback_manager import PlaybackManager
from trace_manager import TraceManager
from audio_manager import AudioManager

# --- ESTADO GLOBAL (Solo cosas de la UI y configuración general) ---
state = {
    'fft_size': 4096,
    'center_freq': 100e6,
    'sample_rate': 10e6,
    'demod_mode': 'none',
    'is_recording': False,
    'recorded_samples': [],
    'zero_span': False
}

class SignalEmitter(QObject):
    # Definimos la señal para actualizar los gráficos desde otros hilos
    data_updated = pyqtSignal(np.ndarray, object, object, object, object, object, object, object, object, object)

emitter = SignalEmitter()

class MainWindow(QMainWindow):
    def __init__(self, radio_handler): 
        super().__init__()
        
        # 1. HARDWARE Y DSP SETUP
        self.radio = radio_handler
        self.radio.rx_callback = self.procesar_muestras_iq # Conectamos la radio a nuestro puente
        self.demodulador_actual = SpectrumAnalyzer() 
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
        
        # 2. CONFIGURACIÓN DE LA VENTANA
        self.setWindowTitle(f"DEMODULADOR SDR - [{self.radio.nombre}]")
        self.resize(QSize(1200, 600))
        self.setMinimumSize(QSize(800, 400))
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        # --- VARIABLES INTERNAS DE LA UI ---
        self.current_freq_multiplier = 1e6
        self.is_paused = False
        self._maximized_widget = None
        self._saved_visibility = {}

        # Iniciamos managers
        self.trace_manager = TraceManager()
        self.audio_manager = AudioManager(self, state)
        self.playback_manager = PlaybackManager(self, state, emitter)

        # Construimos la interfaz gráfica
        self._build_ui()

        # Conectamos el actualizador de gráficos
        emitter.data_updated.connect(self.update_plot)

        # --- Creamos el eje X matemático en memoria ANTES de encender el SDR ---
        self.update_x_axis()
        
        # 3. ENCENDIDO DE LA RADIO
        self.radio.configurar(state['sample_rate'], state['center_freq'])
        self.radio.start_rx()

    def procesar_muestras_iq(self, c_samples):
        # 1. Grabación de muestras I/Q crudas (si el usuario activó la grabación)
        if state['is_recording']:
            state['recorded_samples'].append(c_samples.copy())

        # 2. Procesamiento a través del plugin DSP activo (SpectrumAnalyzer o DemoduladorWBFM)
        if self.demodulador_actual is not None:
            resultados = self.demodulador_actual.procesar(c_samples)
            
            # El plugin devuelve None si todavía está acumulando muestras en su buffer 
            # para cumplir con el bloque de tiempo mínimo (ej: los 100ms de la FM)
            if resultados is not None:
                
                # 3. Gestión de Audio:
                self.audio_manager.enqueue_audio(resultados.get('audio_out'))
                
                # 4. Actualización de la Interfaz: Despachamos todos los vectores procesados 
                # hacia el hilo principal de PyQt usando el emisor de señales genérico.
                emitter.data_updated.emit(
                    resultados['psd_rf'],
                    resultados['rf_chunk'],
                    resultados.get('psd_mpx'),
                    resultados.get('f_axis_mpx'),
                    resultados.get('audio_time_L'), 
                    resultados.get('audio_time_R'), 
                    resultados.get('t_axis_audio'),
                    resultados.get('metricas'),
                    resultados.get('mpx_time'),
                    resultados.get('evm_data')
                )

    # === MÉTODOS DE LA UI (BOTONES Y MENÚS) ===

    def set_wbfm_mode(self):
        if hasattr(self, 'waterfall_checkbox'):
            self.waterfall_checkbox.hide()
            self.waterfall_label.hide()
        if hasattr(self, 'zero_span_btn'):
            self.zero_span_btn.hide()
            self.zero_span_label.hide()
        self.radio.set_muestras_por_bloque(32768)

        # Restauramos el color original de la curva
        self.q3_curve.setPen(pg.mkPen(color="#FF9500", width=1.5))
        self.q3_curve.show()
        if hasattr(self, 'q3_evm_rms_subc'):
            self.q3_evm_rms_subc.hide()
            self.q3_evm_peak_subc.hide()
            self.q3_evm_rms_sym.hide()
            self.q3_evm_peak_sym.hide()
            self.q3_evm_limit.hide()
            if hasattr(self, 'help_q3'): self.help_q3.hide()
            if hasattr(self, 'btn_ideal_const'): self.btn_ideal_const.hide()
            
        # Asegurarnos de que las curvas usen líneas y no puntos de constelación (por si venimos de WiFi)
        self.q4L_curve.setData([], [], pen=pg.mkPen(color="#00FFFF", width=1.5), symbol=None)
        self.q4R_curve.setData([], [], pen=pg.mkPen(color="#FF00FF", width=1.5), symbol=None)
        if hasattr(self, 'q4L_ideal_curve'): self.q4L_ideal_curve.hide()
        if hasattr(self, 'q4L_signal_curve'): self.q4L_signal_curve.setData([], [])
        
        self.audio_container.hide()
        # Configuramos las pantallas
        self.q2_widget.setTitle("Espectro MPX (Audio Demodulado)")
        self.q2_widget.setLabel('bottom', 'Frecuencia [kHz]')
        self.q2_widget.setLabel('left', 'Magnitud [dB]')
        self.q2_widget.setXRange(0, 100)
        self.q2_widget.setYRange(-80, 20)

        self.q3_widget.setTitle("Señal Demodulada en el Tiempo")
        self.q3_widget.getViewBox().invertY(False)
        self.q3_widget.setXLink(None)
        self.q3_widget.setLabel('bottom', 'Tiempo [ms]')
        self.q3_widget.setLabel('left', 'Desviación [kHz]') 
        self.q3_widget.setXRange(0, 10) 
        self.q3_widget.setYRange(-100, 100)

        # Configurar gráfico Canal L
        self.q4L_widget.setTitle("Canal Izquierdo (L)")
        ##self.q4L_widget.hideAxis('bottom') # Ocultamos el eje X de arriba para que quede más limpio
        self.q4L_widget.setLabel('left', 'Amplitud')
        self.q4L_widget.setXRange(0, 10) # 10 ms (mismo tiempo que el MPX)
        self.q4L_widget.setYRange(-1.5, 1.5)
        
        # Configurar gráfico Canal R
        self.q4R_widget.setTitle("Canal Derecho (R)")
        self.q4R_widget.setLabel('bottom', 'Tiempo [ms]')
        self.q4R_widget.setLabel('left', 'Amplitud')
        self.q4R_widget.setXRange(0, 10)
        self.q4R_widget.setYRange(-1.5, 1.5)

        # Mostrar las metricas
        self.fm_metrics_label.show() 
        self.stereo_metrics_label.show()
        self.wifi_metrics_label.hide()
        self.wifi_hw_metrics_label.hide()
        
        self.q2_widget.show()
        self.q3_widget.show()
        self.q4_container.show() # Mostramos el contenedor entero
        self.plot_layout.setRowStretch(1, 1)
        self.plot_layout.setRowStretch(2, 0)
        if hasattr(self, 'q3b_widget'): self.q3b_widget.hide()

        # Ajustamos Sample Rate Visual (Decimado)
        self.previous_sr_text = self.sr_combo.currentText()
        self.sr_combo.blockSignals(True)
        if self.sr_combo.findText("2.4 MHz (Decimado a 300k)") == -1:
            self.sr_combo.addItem("2.4 MHz (Decimado a 300k)")
        self.sr_combo.setCurrentText("2.4 MHz (Decimado a 300k)")
        self.sr_combo.setEnabled(False) 
        self.sr_combo.blockSignals(False)

        # --- CAMBIO DE ARQUITECTURA DSP Y HARDWARE ---
        state['demod_mode'] = 'wbfm'
        state['sample_rate'] = 2.4e6 
        
        # 1. Cargamos el Plugin DSP
        self.demodulador_actual = DemoduladorWBFM()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
        
        # 2. Configuramos la Radio
        self.radio.set_sample_rate(state['sample_rate'])
        
        # 3. Sintonizamos centro de FM
        self.freq_input.setValue(100.0) # Se dispara on_freq_changed
        self.update_x_axis()

    def set_wifi_ag_mode(self):
        if hasattr(self, 'waterfall_checkbox'):
            self.waterfall_checkbox.hide()
            self.waterfall_label.hide()
        if hasattr(self, 'zero_span_btn'):
            self.zero_span_btn.hide()
            self.zero_span_label.hide()
        # 1. Ocultamos las herramientas analógicas
        self.audio_container.hide()
        self.fm_metrics_label.hide()
        self.stereo_metrics_label.hide()
        self.wifi_metrics_label.show()
        self.wifi_hw_metrics_label.show()

        # 2. Mostramos los paneles y expandimos la grilla a 2x2
        self.q2_widget.show()
        self.q3_widget.show()
        self.q4_container.show()
        
        # En Q4 solo necesitamos un gráfico, así que ocultamos el Canal R del estéreo
        self.q4L_widget.show()
        self.q4R_widget.hide() 
        
        self.plot_layout.setRowStretch(0, 1)
        self.plot_layout.setRowStretch(1, 1)
        self.plot_layout.setRowStretch(2, 1)

        # --- CUADRANTE 2: SEÑAL EN EL TIEMPO (I) ---
        self.q2_widget.setTitle("Señal Baseband en el Tiempo")
        self.q2_widget.setLabel('bottom', 'Tiempo [us]')
        self.q2_widget.setLabel('left', 'Amplitud')
        
        self.q2_widget.setXRange(0, 350) 
        self.q2_widget.setYRange(0, 1) # Rango típico de la salida del SDR
        
        # Restauramos la línea continua (quitamos los puntos sueltos que habíamos puesto para la constelación)
        self.q2_curve.setData([], [], pen=pg.mkPen(color="#C3FF00", width=1.5), symbol=None)
        
        # --- CUADRANTE 3: EVM ---
        self.q3_widget.setTitle("EVM por Subportadora")
        self.q3_widget.getViewBox().invertY(False)
        self.q3_widget.setXLink(None)
        self.q3_widget.setLabel('bottom', 'Subportadora')
        self.q3_widget.setLabel('left', 'EVM [dB]')
        self.q3_widget.setXRange(-27, 27)
        self.q3_widget.setYRange(-40, 0)
        
        self.q3b_widget.setTitle("EVM por Símbolo")
        self.q3b_widget.getViewBox().invertY(False)
        self.q3b_widget.setXLink(None)
        self.q3b_widget.setLabel('bottom', 'Símbolo')
        self.q3b_widget.setLabel('left', 'EVM [dB]')
        self.q3b_widget.setYRange(-40, 0)
        
        self.q3_curve.hide()
        
        if hasattr(self, 'q3_evm_rms_subc'):
            self.q3_evm_rms_subc.show()
            self.q3_evm_peak_subc.show()
            self.q3_evm_rms_sym.show()
            self.q3_evm_peak_sym.show()
            self.q3_evm_limit.show()
            self.q3b_evm_limit.show()
            self.help_q3.show()
            self.help_q3b.show()
            self.q3b_widget.show()
            if hasattr(self, 'btn_ideal_const'):
                self.btn_ideal_const.show()

        # --- CUADRANTE 4: CONSTELACIÓN (SÍMBOLOS DE DATOS) ---
        self.q4L_widget.setTitle("Constelación")
        self.q4L_widget.setLabel('bottom', 'En Fase (I)')
        self.q4L_widget.setLabel('left', 'Cuadratura (Q)')
        self.q4L_widget.setXRange(-1.5, 1.5)
        self.q4L_widget.setYRange(-1, 1)
        self.q4L_widget.showGrid(x=False, y=False) # Ocultamos la grilla
        self.q4L_widget.setAspectLocked(True)
        
        # Quitamos la línea (pen=None) y usamos puntos (symbol='o')
        self.q4L_curve.setData([], [], pen=None, symbol='o', symbolSize=5, symbolPen=None, symbolBrush="#00FFFF")

        # --- ARQUITECTURA DSP Y HARDWARE ---
        state['demod_mode'] = 'wifi_ag'
        state['sample_rate'] = 20e6 
        
        # Le decimos a la radio que mande bloques enteros de 2ms (40.000 muestras) y lo redondeamos a una potencia de 2
        if hasattr(self.radio, 'set_muestras_por_bloque'):
            muestras = int(state['sample_rate'] * 0.002)
            # Redondear a la potencia de 2 más cercana hacia arriba
            pot2 = 1
            while pot2 < muestras:
                pot2 *= 2
            self.radio.set_muestras_por_bloque(pot2)
            print(f"[bladeRF] set_muestras_por_bloque: pedido={muestras}, usando={pot2}")

        self.demodulador_actual = DemoduladorWiFiAG()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
        self.radio.set_sample_rate(state['sample_rate'])
        
        # --- Configuración automatica de la frecuencia central al primer canal de wifi
        self.unit_combo.setCurrentText("GHz")
        self.freq_input.setValue(2.412)

        # Ajustamos el combobox visualmente
        self.sr_combo.blockSignals(True)
        if self.sr_combo.findText("20 MHz") != -1:
            self.sr_combo.setCurrentText("20 MHz")
        elif self.sr_combo.findText("20.0 MHz") != -1:
            self.sr_combo.setCurrentText("20.0 MHz")
        self.sr_combo.blockSignals(False)

    def set_wbfm_audio_mode(self):
        self.set_wbfm_mode() 
        self.audio_container.show()
        
        state['demod_mode'] = 'wbfm_audio'
        
        # Cargamos el Plugin de Audio
        self.demodulador_actual = DemoduladorWBFMAudio()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])

    def set_normal_mode(self):
        self.radio.set_muestras_por_bloque(32768)
        self.q2_widget.hide()
        if hasattr(self, 'waterfall_checkbox'):
            self.waterfall_checkbox.show()
            self.waterfall_label.show()
        if hasattr(self, 'zero_span_btn'):
            self.zero_span_btn.show()
            self.zero_span_label.show()
        
        if getattr(self, 'waterfall_enabled', False):
            self.q3_widget.setTitle("Espectrograma (Waterfall)")
            self.q3_widget.setLabel('left', 'Tiempo [ms]')
            self.q3_widget.getViewBox().invertY(True)
            self.q3_widget.setXLink(self.freq_plot)
            self.q3_widget.show()
        else:
            self.q3_widget.setTitle("2-1 (Vacío)")
            self.q3_widget.setLabel('left', '')
            self.q3_widget.getViewBox().invertY(False)
            self.q3_widget.setXLink(None)
            self.q3_widget.hide()
            if hasattr(self, 'q3b_widget'): self.q3b_widget.hide()
            
        self.q4_container.hide() # Ocultamos el contenedor


        # Restauramos el color original de la curva
        self.q3_curve.setPen(pg.mkPen(color="#FF9500", width=1.5))
        self.q3_curve.show()
        if hasattr(self, 'q3_evm_rms_subc'):
            self.q3_evm_rms_subc.hide()
            self.q3_evm_peak_subc.hide()
            self.q3_evm_rms_sym.hide()
            self.q3_evm_peak_sym.hide()
            self.q3_evm_limit.hide()
            if hasattr(self, 'help_q3'): self.help_q3.hide()
            if hasattr(self, 'btn_ideal_const'): self.btn_ideal_const.hide()

        self.plot_layout.setRowStretch(1, 0)
        self.plot_layout.setRowStretch(2, 0)

        # Asegurarnos de que las curvas usen líneas y no puntos de constelación
        self.q4L_curve.setData([], [], pen=pg.mkPen(color="#00FFFF", width=1.5), symbol=None)
        if hasattr(self, 'q4L_ideal_curve'): self.q4L_ideal_curve.hide()
        if hasattr(self, 'q4L_signal_curve'): self.q4L_signal_curve.setData([], [])
        
        # Mostrar el gráfico R que ocultamos en WiFi
        self.q4R_widget.show()
        self.q2_curve.setData([], [], pen=pg.mkPen(color="#C3FF00", width=1.5), symbol=None)
        
        self.q2_widget.setTitle("1-2 (Vacío)")
        self.q2_curve.setData([], []) 
        if not getattr(self, 'waterfall_enabled', False):
            self.q3_widget.setTitle("2-1 (Vacío)")
        self.q3_curve.setData([], [])
        
        # Vaciamos L y R y las metricas
        self.q4L_widget.setTitle("Canal Izquierdo (L) - Vacío")
        self.q4R_widget.setTitle("Canal Derecho (R) - Vacío")
        self.q4L_curve.setData([], [])
        self.q4R_curve.setData([], [])
        self.q4L_signal_curve.setData([], [])
        self.audio_container.hide()
        self.fm_metrics_label.hide()
        self.fm_metrics_label.setText("")
        self.stereo_metrics_label.hide()
        self.stereo_metrics_label.setText("")
        self.wifi_metrics_label.hide()
        self.wifi_metrics_label.setText("")
        self.wifi_hw_metrics_label.hide()
        self.wifi_hw_metrics_label.setText("")
        
        if self.audio_l_btn.isChecked() or self.audio_r_btn.isChecked():
            self.audio_l_btn.setChecked(False)
            self.audio_r_btn.setChecked(False)
            self.audio_manager.toggle_audio()
        
        self.sr_combo.blockSignals(True)
        idx = self.sr_combo.findText("2.4 MHz (Decimado a 300k)")
        if idx != -1: self.sr_combo.removeItem(idx)
        if hasattr(self, 'previous_sr_text'):
            self.sr_combo.setCurrentText(self.previous_sr_text)
        self.sr_combo.setEnabled(True) 
        self.sr_combo.blockSignals(False)
        
        # --- LIMPIEZA DE ARQUITECTURA ---
        state['demod_mode'] = 'none'
        self.demodulador_actual = SpectrumAnalyzer()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
        
        # Restauramos Hardware
        self.on_sr_changed(self.sr_combo.currentText()) 
        self.update_x_axis()

    def on_freq_changed(self, val):
        state['center_freq'] = val * self.current_freq_multiplier
        self.radio.set_freq(state['center_freq'])
        self.update_x_axis()

    def on_sr_changed(self, text):
        if not text: return
        val_mhz = float(text.replace(" MHz", ""))
        state['sample_rate'] = val_mhz * 1e6
        
        self.radio.set_sample_rate(state['sample_rate'])
        
        # Si hay un plugin activo, le avisamos que cambió el sample rate
        if self.demodulador_actual is not None:
            self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
            
        self.update_x_axis()
    
    def on_lna_changed(self, text):
        if not text: return
        val = int(text.replace(" dB", ""))
        self.radio.set_gain(val) # Esto controla el LNA en HackRF o la ganancia global en bladeRF

    def on_vga_changed(self, text):
        if not text: return
        val = int(text.replace(" dB", ""))
        
        # Le preguntamos a la radio si tiene la capacidad de ajustar el VGA
        if hasattr(self.radio, 'set_vga_gain'):
            self.radio.set_vga_gain(val)
        else:
            # Esto es solo un log de seguridad, en la práctica el botón 
            # de VGA ni siquiera va a aparecer si no es una HackRF
            print(f"El equipo {self.radio.nombre} no soporta ajuste de VGA independiente.")

    def on_fft_changed(self, text):
        state['fft_size'] = int(text)
        self.trace_manager.reset()
        if self.demodulador_actual is not None:
            self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
        self.update_x_axis()
            
    def update_spinbox_step(self):
        # Ajusta cuánto salta el valor al apretar las flechitas según la unidad
        if self.current_freq_multiplier == 1:       # Hz
            self.freq_input.setSingleStep(10000.0)
            self.freq_input.setDecimals(0)
        elif self.current_freq_multiplier == 1e3:   # kHz
            self.freq_input.setSingleStep(10.0)
            self.freq_input.setDecimals(3)
        elif self.current_freq_multiplier == 1e6:   # MHz
            self.freq_input.setSingleStep(0.1)
            self.freq_input.setDecimals(6)
        elif self.current_freq_multiplier == 1e9:   # GHz
            self.freq_input.setSingleStep(0.0001)
            self.freq_input.setDecimals(9)

    def on_unit_changed(self, unit_text):
        # Diccionario con los multiplicadores
        multipliers = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
        new_multiplier = multipliers[unit_text]
        
        # Bloqueamos las señales temporalmente para que cambiar la vista 
        # no mande comandos locos a la SDR
        self.freq_input.blockSignals(True)
        
        # Calculamos cómo se tiene que ver la frecuencia absoluta en la nueva unidad
        display_val = state['center_freq'] / new_multiplier
        self.current_freq_multiplier = new_multiplier
        
        self.update_spinbox_step()
        self.freq_input.setValue(display_val)
        
        # Volvemos a habilitar las señales
        self.freq_input.blockSignals(False)

    def update_x_axis(self):
        cf = state['center_freq']
        if state.get('demod_mode') == 'wbfm':
            fs = state['fft_size'] 
            sr_visual = 300000 
            self.f_axis = np.linspace(cf - sr_visual/2, cf + sr_visual/2, fs) / 1e6
            self.freq_plot.setXRange((cf - sr_visual/2)/1e6, (cf + sr_visual/2)/1e6)
        else:
            sr = state['sample_rate']
            fs = state['fft_size']
            self.f_axis = np.linspace(cf - sr/2, cf + sr/2, fs) / 1e6
            self.freq_plot.setXRange((cf - sr/2)/1e6, (cf + sr/2)/1e6)
    

    def closeEvent(self, event):
        print("Cerrando aplicación SDR...")
        self.audio_manager.stop_all()
        self.radio.close()
        event.accept()
        
    def keyPressEvent(self, event):
        # Escape restaura el panel maximizado
        if event.key() == Qt.Key.Key_Escape and self._maximized_widget is not None:
            self._restore_panels()
            return

        # Pausa con el espacio
        if event.key() == Qt.Key.Key_Space:
            self.pause_btn.click() # Simula el click físico en el botón
            return # Cortamos la ejecución acá para que no haga nada más

        # Le pasamos el evento al manager. Si lo pudo procesar (porque era una flechita),
        # devuelve True. Si devolvió False, dejamos que la ventana haga lo suyo por defecto.
        handled = self.marker_manager.handle_key_press(
            event, 
            self.f_axis, 
            getattr(self, 'last_f_axis_audio', None)
        )
        if not handled:
            super().keyPressEvent(event)

    # --- SISTEMA DE MAXIMIZAR/RESTAURAR PANELES (doble-click) ---
    def eventFilter(self, obj, event):
        if event.type() == event.Type.MouseButtonDblClick:
            if self._maximized_widget is None:
                self._maximize_panel(obj)
            else:
                self._restore_panels()
            return True
        return super().eventFilter(obj, event)

    def _maximize_panel(self, widget):
        """Oculta los demás paneles y expande la celda del seleccionado para que ocupe toda la grilla."""
        all_panels = [self.freq_plot, self.q2_widget, self.q3_widget, 
                      self.q3b_widget, self.q4_container]
        
        # Guardar estado original
        self._saved_visibility = {w: w.isVisible() for w in all_panels}
        self._saved_row_stretches = {i: self.plot_layout.rowStretch(i) for i in range(self.plot_layout.rowCount())}
        self._saved_col_stretches = {i: self.plot_layout.columnStretch(i) for i in range(self.plot_layout.columnCount())}
        
        # Encontrar dónde está el widget en la grilla
        idx = self.plot_layout.indexOf(widget)
        if idx == -1: return
        row, col, rowSpan, colSpan = self.plot_layout.getItemPosition(idx)
        
        # Ocultar todos menos el seleccionado
        for w in all_panels:
            if w is not widget:
                w.hide()
        widget.show()
        
        # Darle todo el stretch a la fila/columna de nuestro widget, y 0 al resto
        for i in range(self.plot_layout.rowCount()):
            self.plot_layout.setRowStretch(i, 1 if (row <= i < row + rowSpan) else 0)
        for i in range(self.plot_layout.columnCount()):
            self.plot_layout.setColumnStretch(i, 1 if (col <= i < col + colSpan) else 0)
            
        self._maximized_widget = widget

    def _restore_panels(self):
        """Restaura todos los paneles a sus posiciones y tamaños originales."""
        if self._maximized_widget is None: return
        
        # Restaurar visibilidad
        for w, was_visible in self._saved_visibility.items():
            w.setVisible(was_visible)
            
        # Restaurar stretches
        for i, s in self._saved_row_stretches.items():
            self.plot_layout.setRowStretch(i, s)
        for i, s in self._saved_col_stretches.items():
            self.plot_layout.setColumnStretch(i, s)
            
        self._maximized_widget = None
        self._saved_visibility = {}

    def toggle_pause(self):
        self.is_paused = self.pause_btn.isChecked()
        
        if self.is_paused:
            self.pause_btn.setText("▶")
            self.pause_btn.setStyleSheet("background-color: #488BD8; color: black; font-size: 16px; font-weight: bold; border-radius: 4px; margin: 4px;")
        else:
            self.pause_btn.setText("⏸")
            self.pause_btn.setStyleSheet("background-color: #444; color: white; font-size: 16px; font-weight: bold; border-radius: 4px; margin: 4px;")

    def toggle_zero_span(self):
        state['zero_span'] = self.zero_span_btn.isChecked()
        
        if state['zero_span']:
            # Botón activo (Azul)
            self.zero_span_btn.setStyleSheet("background-color: #0077FF; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #0055FF;")
            self.freq_plot.setLabel('bottom', 'Tiempo [us]')
        else:
            # Botón inactivo (Gris)
            self.zero_span_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #555;")
            self.freq_plot.setLabel('bottom', 'Frecuencia [MHz]')
            self.update_x_axis() # Restaura la vista de frecuencia

    def on_waterfall_toggled(self, state_val):
        self.waterfall_enabled = self.waterfall_checkbox.isChecked()
        if self.waterfall_enabled:
            # Restablecer el buffer si se activa para no mostrar ruido viejo
            self.waterfall_buffer = None
            if state.get('demod_mode') == 'none':
                self.q3_widget.setTitle("Espectrograma (Waterfall)")
                self.q3_widget.setLabel('left', 'Tiempo [ms]')
                self.q3_widget.enableAutoRange(axis='xy')
                self.q3_widget.getViewBox().invertY(True)
                self.q3_widget.setXLink(self.freq_plot)
                self.q3_widget.show()
        else:
            if state.get('demod_mode') == 'none':
                self.q3_widget.setTitle("2-1 (Vacío)")
                self.q3_widget.setLabel('left', '')
                self.q3_widget.getViewBox().invertY(False)
                self.q3_widget.setXLink(None)
                # Limpiar el ImageItem
                self.waterfall_image.clear()
                self.q3_widget.hide()

    def update_plot(self, PSD, raw_samples, PSD_audio=None, f_axis_audio=None, audio_L=None, audio_R=None, t_axis=None, fm_metrics=None, mpx_time=None, evm_data=None):
        if self.is_paused:
            return
        
        if len(self.f_axis) == len(PSD):
            
            # --- LÓGICA DE TRACE ---
            if state.get('zero_span', False):
                if raw_samples is not None:
                    # 1. Calculamos la envolvente de potencia en el tiempo (dB)
                    # |I + jQ|^2 nos da la potencia. Sumamos 1e-12 para evitar log(0)
                    power_time = 10.0 * np.log10(np.abs(raw_samples)**2 + 1e-12)
                    
                    # 2. Generamos el eje de tiempo en microsegundos (us)
                    t_axis = (np.arange(len(raw_samples)) / state['sample_rate']) * 1e6
                    
                    # 3. Graficamos la señal temporal
                    self.freq_plot_curve.setData(t_axis, power_time)
                    
                    # Fijamos la ventana temporal para que no baile el gráfico
                    self.freq_plot.setXRange(0, t_axis[-1])
                    
                    # Nota: En Zero Span no llamamos a self.marker_manager.update_render
                    # porque los markers están atados a coordenadas de frecuencia.
            else:
                # FLUJO NORMAL (Dominio de Frecuencia)
                if len(self.f_axis) == len(PSD):
                    # --- LÓGICA DE TRACE ---
                    display_psd = self.trace_manager.process(PSD)
                    self.freq_plot_curve.setData(self.f_axis, display_psd)
                    
                    # --- LÓGICA DE WATERFALL ---
                    if getattr(self, 'waterfall_enabled', False) and state.get('demod_mode') == 'none':
                        if self.waterfall_buffer is None or self.waterfall_buffer.shape[0] != len(PSD):
                            self.waterfall_lines = 200
                            self.waterfall_buffer = np.zeros((len(PSD), self.waterfall_lines))
                            self.waterfall_buffer.fill(-130)
                            self.waterfall_counter = 0
                        
                        self.waterfall_counter = getattr(self, 'waterfall_counter', 0) + 1
                        
                        if self.waterfall_counter >= 7:
                            self.waterfall_counter = 0
                            
                            # Desplazamos las columnas de tiempo hacia la derecha (o izquierda, según el waterfall)
                            self.waterfall_buffer = np.roll(self.waterfall_buffer, 1, axis=1)
                            self.waterfall_buffer[:, 0] = display_psd
                            
                            # Actualizamos la imagen (X: frecuencia, Y: tiempo)
                            # autoLevels=False y levels fijos evitan que pyqtgraph colapse calculando
                            # los min/max de la matriz enorme docenas de veces por segundo.
                            self.waterfall_image.setImage(
                                self.waterfall_buffer, 
                                autoLevels=False, 
                                levels=(-130, -30), 
                                autoDownsample=True
                            )
                            
                            # Ajustamos la escala para que coincida con el eje X de frecuencias
                            f_min, f_max = self.f_axis[0], self.f_axis[-1]
                            
                            if raw_samples is not None and state.get('sample_rate'):
                                block_time_ms = (len(raw_samples) / state['sample_rate']) * 1000.0
                                total_time_ms = block_time_ms * 7 * self.waterfall_lines
                            else:
                                total_time_ms = self.waterfall_lines * 7
                                
                            self.waterfall_image.setRect(pg.QtCore.QRectF(f_min, 0, f_max - f_min, total_time_ms))

                    # --- LÓGICA DE MARKERS Y DELTAS ---
                    self.marker_manager.update_render(
                        display_psd, 
                        self.f_axis, 
                        PSD_audio, 
                        getattr(self, 'last_f_axis_audio', None), 
                        state.get('demod_mode')
                    )

            # --- RENDERIZADO ESPECÍFICO DE WIFI A/G ---
            if state.get('demod_mode') == 'wifi_ag':
                if raw_samples is not None:
                    # 1. Generamos el eje de tiempo en microsegundos (us)
                    t_axis_us = (np.arange(len(raw_samples)) / state['sample_rate']) * 1e6
                    
                    # 2. Graficamos la envolvente (Magnitud)
                    signal_env = np.abs(raw_samples)
                    self.q2_curve.setData(t_axis_us, signal_env)
                
                # --- GRÁFICAMOS EVM EN EL Q3 ---
                if evm_data is not None and hasattr(self, 'q3_evm_rms_subc'):
                    y_floor = -40
                    # Barras crecen hacia arriba desde el piso del gráfico (como en el notebook)
                    subc_rms_db = np.asarray(evm_data['subc_rms'])
                    subc_peak_db = np.asarray(evm_data['subc_peak'])
                    
                    # Coloreamos distinto a las subportadoras piloto (-21, -7, 7, 21)
                    brushes_peak = [pg.mkBrush(180, 50, 255, 150) if x in [-21, -7, 7, 21] else pg.mkBrush(100, 100, 255, 100) for x in evm_data['subc_x']]
                    brushes_rms = [pg.mkBrush(140, 0, 220, 200) if x in [-21, -7, 7, 21] else pg.mkBrush(0, 0, 150, 200) for x in evm_data['subc_x']]
                    
                    self.q3_evm_peak_subc.setOpts(x=evm_data['subc_x'], height=subc_peak_db - y_floor, y0=y_floor, brushes=brushes_peak)
                    self.q3_evm_rms_subc.setOpts(x=evm_data['subc_x'], height=subc_rms_db - y_floor, y0=y_floor, brushes=brushes_rms)
                    
                    n_syms = len(evm_data['sym_rms'])
                    if n_syms > 0:
                        sym_x = np.arange(n_syms)
                        self.q3_evm_rms_sym.setData(sym_x, evm_data['sym_rms'])
                        self.q3_evm_peak_sym.setData(sym_x, evm_data['sym_peak'])
                        self.q3b_widget.setXRange(0, max(n_syms - 1, 1))
                    else:
                        self.q3_evm_rms_sym.setData([], [])
                        self.q3_evm_peak_sym.setData([], [])
                        
                    metrics = fm_metrics.get('wifi_metrics', {}) if fm_metrics else {}
                    mod = metrics.get('mod', '')
                    mbps = metrics.get('mbps', '')
                    if mod:
                        # Límites EVM del estándar IEEE 802.11a/g (Table 17-10)
                        EVM_LIMITS_DB = {
                            6: -5, 9: -8, 12: -10, 18: -13,
                            24: -16, 36: -19, 48: -22, 54: -25
                        }
                        limite_db = EVM_LIMITS_DB.get(int(mbps), -25)
                        self.q3_evm_limit.setPos(limite_db)
                        self.q3b_evm_limit.setPos(limite_db)
                        self.q3_widget.setTitle(f"EVM por Subportadora ({mod} / {mbps} Mbps) — Límite: {limite_db} dB")
                        self.q3b_widget.setTitle(f"EVM por Símbolo ({mod} / {mbps} Mbps) — Límite: {limite_db} dB")
                elif mpx_time is not None and not hasattr(self, 'q3_evm_rms_subc'):
                    eje_x_sc = np.arange(len(mpx_time))
                    self.q3_curve.setData(eje_x_sc, mpx_time)

                # --- GRAFICAMOS LAS CONSTELACIÓNES EN Q4 ---
                # Recibimos las coordenadas X(I) e Y(Q) a través de los canales de audio
                if audio_L is not None and audio_R is not None:
                    # 1. Pasamos nuevamente los parámetros del símbolo para refrescar el ScatterPlotItem
                    self.q4L_curve.setData(
                        audio_L, 
                        audio_R, 
                        pen=None, 
                        symbol='o', 
                        symbolSize=2.5, 
                        symbolPen=None, 
                        symbolBrush="#00FFFF"
                    )
                if PSD_audio is not None and f_axis_audio is not None:
                    self.q4L_signal_curve.setData(
                        PSD_audio, 
                        f_axis_audio, 
                        pen=None, 
                        symbol='o', 
                        symbolSize=2.5, 
                        symbolPen="#FF00FF", 
                        symbolBrush=None
                    )
                    
                    # 2. Forzamos a Qt a repintar la escena del widget de forma manual e inmediata
                    self.q4L_widget.scene().update()
                
                # Actualizar Constelación Ideal si el botón está activado
                mod = fm_metrics.get('wifi_metrics', {}).get('mod', '') if fm_metrics else ''
                if mod and hasattr(self, 'btn_ideal_const') and self.btn_ideal_const.isChecked():
                    if mod == 'BPSK': niveles = [-1, 1]
                    elif mod == 'QPSK': niveles = [-1, 1]
                    elif mod == '16-QAM': niveles = [-3, -1, 1, 3]
                    elif mod == '64-QAM': niveles = [-7, -5, -3, -1, 1, 3, 5, 7]
                    else: niveles = []
                    
                    if niveles:
                        import itertools
                        pts = [(x, 0) for x in niveles] if mod == 'BPSK' else list(itertools.product(niveles, niveles))
                        pts_arr = np.array(pts, dtype=float)
                        multiplicador = 1 if mod == 'BPSK' else 2
                        P_teorica = np.mean(np.array(niveles)**2) * multiplicador
                        escala = np.sqrt(1.0 / P_teorica)
                        pts_arr *= escala
                        self.q4L_ideal_curve.setData(pts_arr[:,0], pts_arr[:,1])
                        self.q4L_ideal_curve.show()
                    else:
                        self.q4L_ideal_curve.hide()
                else:
                    if hasattr(self, 'q4L_ideal_curve'):
                        self.q4L_ideal_curve.hide()
            

            if state.get('demod_mode') in ['wbfm', 'wbfm_audio']:
                if PSD_audio is not None:
                    self.q2_curve.setData(f_axis_audio, PSD_audio)
                if mpx_time is not None:                              
                    self.q3_curve.setData(t_axis, mpx_time)
                if audio_L is not None and audio_R is not None:
                    self.q4L_curve.setData(t_axis, audio_L)
                    self.q4R_curve.setData(t_axis, audio_R)
                
                # Renderizar las métricas en HTML en el panel derecho
                if fm_metrics is not None:
                    html_text = (
                        f"<div style='line-height: 1.5;'>"
                        f"<span style='color: #FFFFFF'><b>Desv. Pico Máx:</b></span> <span style='color: #00B000;'>{fm_metrics['pico_max']:+.2f} kHz</span><br>"
                        f"<span style='color: #FFFFFF'><b>Desv. Pico Mín:</b></span> <span style='color: #FF3333;'>{fm_metrics['pico_min']:+.2f} kHz</span><br>"
                        f"<span style='color: #FFFFFF'><b>Desv. Pico RMS:</b></span> <span style='color: #FFD500;'>{fm_metrics['pico_rms']:.2f} kHz</span><br>"
                        f"<span style='color: #FFFFFF'><b>RMS (True):</b></span> <span style='color: #0077FF;'>{fm_metrics['rms']:.2f} kHz</span><br>"
                        f"<span style='color: #FFFFFF'><b>DC Offset:</b></span> <span style='color: #00FFFF;'>{fm_metrics['dc_offset']:+.2f} kHz</span>"
                        f"</div>"
                    )
                    self.fm_metrics_label.setText(html_text)
                    
                # ---  Cálculo y renderizado de Separación Estéreo ---
                    if 'rms_L' in fm_metrics and 'rms_R' in fm_metrics:
                        # Protegemos el logaritmo por si hay silencio absoluto
                        rms_L_val = max(fm_metrics['rms_L'], 1e-12)
                        rms_R_val = max(fm_metrics['rms_R'], 1e-12)
                        
                        db_L = 20 * np.log10(rms_L_val)
                        db_R = 20 * np.log10(rms_R_val)
                        
                        # La separación es la diferencia absoluta de energía
                        separacion_db = abs(db_L - db_R)
                        
                        # Código de colores (50dB+ es excelente para grado laboratorio)
                        if separacion_db >= 50:
                            color_sep = "#00B000" # Verde
                        elif separacion_db >= 30:
                            color_sep = "#FFD500" # Amarillo
                        else:
                            color_sep = "#FF3333" # Rojo (Música normal o mal filtrado)
                            
                        html_stereo = (
                            f"<div style='line-height: 1.5;'>"
                            f"<span style='color: #FFFFFF'><b>Potencia RMS (L):</b></span> <span style='color: #00FFFF;'>{db_L:+.2f} dB</span><br>"
                            f"<span style='color: #FFFFFF'><b>Potencia RMS (R):</b></span> <span style='color: #FF00FF;'>{db_R:+.2f} dB</span><br>"
                            f"<span style='color: #555555;'>─────────────────────────────</span><br>"
                            f"<span style='color: #FFFFFF'><b>Crosstalk (Separación):</b></span> "
                            f"<span style='color: {color_sep}; font-weight: bold;'>{separacion_db:.2f} dB</span>"
                            f"</div>"
                        )
                        self.stereo_metrics_label.setText(html_stereo)

            if state.get('demod_mode') == 'wifi_ag':
                if fm_metrics and 'wifi_metrics' in fm_metrics:
                    wifi = fm_metrics['wifi_metrics']
                    if wifi is not None:
                        paridad_ok = wifi.get('paridad_ok', False)
                        color_paridad = "#00B000" if paridad_ok else "#FF3333"
                        texto_paridad = "Ok" if paridad_ok else "ERROR"
                        html_wifi = (
                            f"<div style='line-height: 1.5;'>"
                            f"<span style='color: #FFFFFF'><b>Rate Code:</b></span> <span style='color: #00FFFF;'>{wifi.get('rate_code', '?')}</span><br>"
                            f"<span style='color: #FFFFFF'><b>Modulación:</b></span> <span style='color: #FFD500;'>{wifi.get('mod', '?')}</span><br>"
                            f"<span style='color: #FFFFFF'><b>Code rate:</b></span> <span style='color: #FFD500;'>{wifi.get('code_rate', '?')}</span><br>"
                            f"<span style='color: #FFFFFF'><b>Data rate:</b></span> <span style='color: #00FFFF;'>{wifi.get('mbps', 0)} mbps</span><br>"
                            f"<span style='color: #FFFFFF'><b>Length:</b></span> <span style='color: #00FFFF;'>{wifi.get('length', 0)}</span><br>"
                            f"<span style='color: #FFFFFF'><b>Paridad:</b></span> <span style='color: {color_paridad}; font-weight: bold;'>{texto_paridad}</span>"
                            f"</div>"
                        )
                        self.wifi_metrics_label.setText(html_wifi)
                        
                        cfo = wifi.get('cfo', 0.0)
                        cfo_fino = wifi.get('cfo_fino', 0.0)
                        snr = wifi.get('snr', 0.0)
                        html_hw = (
                            f"<div style='line-height: 1.5;'>"
                            f"<span style='color: #FFFFFF'><b>CFO Grueso:</b></span> <span style='color: #FF5555;'>{cfo:+.1f} Hz</span><br>"
                            f"<span style='color: #FFFFFF'><b>CFO Fino:</b></span> <span style='color: #FF5555;'>{cfo_fino:+.1f} Hz</span><br>"
                            f"<span style='color: #FFFFFF'><b>SNR:</b></span> <span style='color: #55FF55;'>{snr:.1f} dB</span>"
                            f"</div>"
                        )
                        self.wifi_hw_metrics_label.setText(html_hw)
    
    def _build_ui(self):
        # --- BARRA SUPERIOR  ---
        self.toolbar = QToolBar("Barra Principal")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.toolbar.setMovable(False)

        # 1 Rec/Play
        self.rec_play_btn = QToolButton()
        self.rec_play_btn.setText("Rec/Play")
        self.rec_play_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.rec_play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.rec_play_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")

        # Botón Pausa/Reanudar
        self.pause_btn = QToolButton()
        self.pause_btn.setText("⏸")
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_btn.setCheckable(True)
        self.pause_btn.setFixedSize(QSize(45, 40))
        self.pause_btn.setStyleSheet("background-color: #444; color: white; font-size: 16px; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.toolbar.addWidget(self.pause_btn)

        # 2. Crear el Menú que va a contener las opciones
        self.rec_play_menu = QMenu()
        self.rec_play_menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
            }
            QMenu::item:selected {
                background-color: #555555;
            }
        """)

        # 3. Crear las acciones (Opciones del menú)
        self.record_action = QAction("🔴 Iniciar Grabación", self)
        self.record_action.triggered.connect(self.playback_manager.toggle_recording)
        
        self.play_action = QAction(" ▶ Reproducir Archivo", self)
        self.play_action.triggered.connect(lambda: self.playback_manager.load_and_play(loop=False))
        
        # Boton de reproducir en loop y de detener
        self.loop_action = QAction("🔁 Reproducir archivo en loop", self)
        self.loop_action.triggered.connect(lambda: self.playback_manager.load_and_play(loop=True))

        self.stop_play_action = QAction("⏹ Detener Reproducción", self)
        self.stop_play_action.triggered.connect(self.playback_manager.stop_playback)
        self.stop_play_action.setEnabled(False) # Arranca deshabilitado
        
        # 4. Agregar al menú
        self.rec_play_menu.addAction(self.record_action)
        self.rec_play_menu.addAction(self.play_action)
        self.rec_play_menu.addAction(self.loop_action)
        self.rec_play_menu.addSeparator() # Una rayita separadora queda linda
        self.rec_play_menu.addAction(self.stop_play_action)

        self.rec_play_btn.setMenu(self.rec_play_menu)
        self.toolbar.addWidget(self.rec_play_btn)

        self.toolbar.addSeparator() # Una barrita vertical para separar

        main_layout = QHBoxLayout()
    
        # --- LADO IZQUIERDO: GRÁFICO ---
        self.freq_plot = pg.PlotWidget(labels={'left': 'Potencia [dB]', 'bottom': 'Frecuencia [MHz]'})
        self.freq_plot.setMouseEnabled(x=True, y=True)
        self.freq_plot.setYRange(-130, 10)
        self.freq_plot_curve = self.freq_plot.plot([], pen=pg.mkPen(color='#FFD500', width=1.5))

        # --- CREACIÓN DE CUADRANTES  ---
        self.q2_widget = pg.PlotWidget(title="1-2 (Vacío)")
        self.q3_widget = pg.PlotWidget(title="2-1 (Vacío)")

        # --- Cuadrante 4 (L y R apilados) ---
        self.q4_container = QWidget()
        self.q4_layout = QVBoxLayout(self.q4_container)
        self.q4_layout.setContentsMargins(0, 0, 0, 0)
        self.q4_layout.setSpacing(2)

        self.q4L_widget = pg.PlotWidget(title="Canal Izquierdo (L)")
        self.q4R_widget = pg.PlotWidget(title="Canal Derecho (R)")
        self.q4R_widget.setXLink(self.q4L_widget)
        self.q4R_widget.setYLink(self.q4L_widget)
        self.q4_layout.addWidget(self.q4L_widget)
        self.q4_layout.addWidget(self.q4R_widget)

        # --- SISTEMA DE MARKERS ---
        self.marker_manager = MarkerManager(self, state['center_freq']/1e6)
        self.marker_manager.attach_to_plots()

        # --- CONTENEDOR GRILLA (2x2) ---
        self.plot_container = QWidget()
        self.plot_layout = QGridLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(5) 
        self.plot_layout.setRowStretch(0, 1) 
        self.plot_layout.setRowStretch(1, 0) 
        self.plot_layout.setRowStretch(2, 0) 

        # Agregamos a la grilla principal
        self.plot_layout.addWidget(self.freq_plot, 0, 0)
        self.plot_layout.addWidget(self.q2_widget, 0, 1) 
        self.plot_layout.addWidget(self.q3_widget, 1, 0) 
        
        self.q3b_widget = pg.PlotWidget(title="EVM por Símbolo")
        self.plot_layout.addWidget(self.q3b_widget, 2, 0)
        self.q3b_widget.hide()
        
        self.plot_layout.addWidget(self.q4_container, 1, 1, 2, 1) 

        # Dejamos las curvas creadas
        self.q2_curve = self.q2_widget.plot([], pen=pg.mkPen(color="#C3FF00", width=1.5))
        self.q3_curve = self.q3_widget.plot([], pen=pg.mkPen(color="#FF9500", width=1.5))
        self.q4L_curve = self.q4L_widget.plot([], pen=pg.mkPen(color="#00FFFF", width=1.5))
        self.q4R_curve = self.q4R_widget.plot([], pen=pg.mkPen(color="#FF00FF", width=1.5))
        self.q4L_signal_curve = self.q4L_widget.plot([], pen=None, symbol='o', symbolSize=2.5, symbolPen="#FF00FF")
        
        cross_path = QPainterPath()
        cross_path.moveTo(-0.5, 0)
        cross_path.lineTo(0.5, 0)
        cross_path.moveTo(0, -0.5)
        cross_path.lineTo(0, 0.5)
        
        self.q4L_ideal_curve = self.q4L_widget.plot([], pen=None, symbol=cross_path, symbolSize=30, symbolPen=pg.mkPen(color="#606060", width=1), symbolBrush=None)

        # --- WATERFALL (Espectrograma) ---
        self.waterfall_image = pg.ImageItem()
        self.waterfall_colormap = pg.colormap.get('viridis')
        self.waterfall_image.setLookupTable(self.waterfall_colormap.getLookupTable())
        self.q3_widget.addItem(self.waterfall_image)
        self.waterfall_enabled = False
        self.waterfall_buffer = None


        # --- NUEVOS ELEMENTOS EVM PARA Q3 ---
        self.q3_evm_peak_subc = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush(100, 100, 255, 100))
        self.q3_evm_rms_subc = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush(0, 0, 150, 200))
        self.q3_evm_rms_sym = self.q3b_widget.plot([], pen=pg.mkPen(color="#00FF00", width=2))
        self.q3_evm_peak_sym = self.q3b_widget.plot([], pen=pg.mkPen(color="#FF3333", width=1.5, style=Qt.PenStyle.DashLine))
        self.q3_evm_limit = pg.InfiniteLine(pos=-25, angle=0, pen=pg.mkPen(color="#FFFFFF", style=Qt.PenStyle.DashLine))
        self.q3b_evm_limit = pg.InfiniteLine(pos=-25, angle=0, pen=pg.mkPen(color="#FFFFFF", style=Qt.PenStyle.DashLine))
        
        self.q3_widget.addItem(self.q3_evm_peak_subc)
        self.q3_widget.addItem(self.q3_evm_rms_subc)
        self.q3_widget.addItem(self.q3_evm_limit)
        self.q3b_widget.addItem(self.q3b_evm_limit)
        
        self.q3_evm_rms_subc.hide()
        self.q3_evm_peak_subc.hide()
        self.q3_evm_rms_sym.hide()
        self.q3_evm_peak_sym.hide()
        self.q3_evm_limit.hide()
        self.q3b_evm_limit.hide()

        # --- ICONOS DE AYUDA (EVM) ---
        # Panel Q3
        self.help_q3 = QLabel("?", self.q3_widget)
        self.help_q3.setStyleSheet("background-color: rgba(60, 60, 60, 200); color: white; border-radius: 10px; font-weight: bold;")
        self.help_q3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.help_q3.resize(20, 20)
        self.help_q3.move(10, 10)
        
        self.tooltip_q3 = QLabel("Azul oscuro: EVM RMS de la subportadora\nCeleste: EVM Pico de la subportadora\nVioleta: Portadoras\nBlanco: Límite del estándar", self.q3_widget)
        self.tooltip_q3.setStyleSheet("background-color: rgba(30, 30, 30, 240); color: white; border: 1px solid #777; padding: 8px; border-radius: 5px; font-size: 14px;")
        self.tooltip_q3.adjustSize()
        self.tooltip_q3.move(35, 10)
        self.tooltip_q3.hide()
        
        self.help_q3.enterEvent = lambda e: self.tooltip_q3.show()
        self.help_q3.leaveEvent = lambda e: self.tooltip_q3.hide()
        self.help_q3.hide()

        # Panel Q3b
        self.help_q3b = QLabel("?", self.q3b_widget)
        self.help_q3b.setStyleSheet("background-color: rgba(60, 60, 60, 200); color: white; border-radius: 10px; font-weight: bold;")
        self.help_q3b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.help_q3b.resize(20, 20)
        self.help_q3b.move(10, 10)
        
        self.tooltip_q3b = QLabel("Verde: EVM RMS de cada símbolo OFDM\nRojo punteado: EVM Pico de cada símbolo\nBlanco: Límite del estándar", self.q3b_widget)
        self.tooltip_q3b.setStyleSheet("background-color: rgba(30, 30, 30, 240); color: white; border: 1px solid #777; padding: 8px; border-radius: 5px; font-size: 14px;")
        self.tooltip_q3b.adjustSize()
        self.tooltip_q3b.move(35, 10)
        self.tooltip_q3b.hide()
        
        self.help_q3b.enterEvent = lambda e: self.tooltip_q3b.show()
        self.help_q3b.leaveEvent = lambda e: self.tooltip_q3b.hide()
        self.help_q3b.hide()

        # Botón Constelación Ideal en Q4L
        self.btn_ideal_const = QPushButton("⌖", self.q4L_widget)
        self.btn_ideal_const.setStyleSheet("QPushButton { background-color: rgba(60, 60, 60, 200); color: white; border-radius: 10px; font-weight: bold; font-size: 16px; border: none; } QPushButton:checked { background-color: rgba(200, 200, 200, 220); color: black; }")
        self.btn_ideal_const.setCheckable(True)
        self.btn_ideal_const.resize(24, 24)
        self.btn_ideal_const.move(10, 10)
        self.btn_ideal_const.setToolTip("Mostrar Constelación Ideal")
        self.btn_ideal_const.hide()

        # Ocultamos por defecto
        self.q2_widget.hide()
        self.q3_widget.hide()
        self.q4_container.hide()

        # --- Instalamos event filters para doble-click maximizar ---
        for w in [self.freq_plot, self.q2_widget, self.q3_widget, self.q3b_widget, self.q4_container]:
            w.installEventFilter(self)

        # Agregamos el contenedor entero al layout principal
        main_layout.addWidget(self.plot_container, stretch=4)

        # --- MENÚ DE MARKERS ---
        self.markers_btn = QToolButton()
        self.markers_btn.setText("Markers")
        self.markers_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.markers_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.markers_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")

        self.markers_menu = QMenu()
        self.markers_menu.setStyleSheet("""
            QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
            QMenu::item:selected { background-color: #555555; }
        """)

        # Grupo de acciones (Radio buttons para elegir qué marker mover)
        self.marker_group = QActionGroup(self)
        self.marker_group.setExclusive(True)

        def create_marker_action(text, key):
            action = QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, k=key: self.marker_manager.select_marker(k))
            self.marker_group.addAction(action)
            self.markers_menu.addAction(action)
            return action

        self.action_m1 = create_marker_action("📍 Seleccionar M1", 'M1')
        self.action_d1 = create_marker_action("📍 Seleccionar Delta 1", 'D1')
        self.action_m2 = create_marker_action("📍 Seleccionar M2", 'M2')
        self.action_d2 = create_marker_action("📍 Seleccionar Delta 2", 'D2')

        self.markers_menu.addSeparator()
        
        self.action_none = QAction("🚫 Mover Ninguno", self)
        self.action_none.setCheckable(True)
        self.action_none.setChecked(True) # Por defecto no se mueve ninguno
        self.action_none.triggered.connect(lambda: self.marker_manager.select_marker(None))
        self.marker_group.addAction(self.action_none)
        self.markers_menu.addAction(self.action_none)

        self.markers_menu.addSeparator()
        
        # --- SUBMENÚ DE ELIMINACIÓN ---
        self.delete_menu = QMenu("🗑️ Eliminar...", self)
        self.delete_menu.setStyleSheet("""
            QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
            QMenu::item:selected { background-color: #8b0000; } /* Un rojito oscuro al seleccionar */
        """)

        def create_delete_action(text, key):
            action = QAction(text, self)
            action.triggered.connect(lambda checked, k=key: self.marker_manager.delete_marker(k))
            self.delete_menu.addAction(action)
            return action

        create_delete_action("❌ Eliminar M1", 'M1')
        create_delete_action("❌ Eliminar Delta 1", 'D1')
        create_delete_action("❌ Eliminar M2", 'M2')
        create_delete_action("❌ Eliminar Delta 2", 'D2')
        
        self.delete_menu.addSeparator()
        
        self.clear_markers_action = QAction("💥 Limpiar Todos", self)
        self.clear_markers_action.triggered.connect(self.marker_manager.clear_markers)
        self.delete_menu.addAction(self.clear_markers_action)

        # Agregar el submenú al menú principal
        self.markers_menu.addMenu(self.delete_menu)

        self.markers_btn.setMenu(self.markers_menu)
        self.toolbar.addWidget(self.markers_btn)
    

        # --- MENÚ DE DEMODULACIONES ---
        self.demod_btn = QToolButton()
        self.demod_btn.setText("Demodulación")
        self.demod_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.demod_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.demod_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")

        self.demod_menu = QMenu()
        self.demod_menu.setStyleSheet("""
            QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
            QMenu::item:selected { background-color: #555555; }
        """)

        # Grupo de acciones para que solo una demodulación esté activa a la vez en toda la app
        self.demod_group = QActionGroup(self)
        self.demod_group.setExclusive(True)

        # Acción: Sin Demodular (Por defecto)
        self.action_demod_none = QAction("Sin Demodular", self)
        self.action_demod_none.setCheckable(True)
        self.action_demod_none.setChecked(True)
        self.action_demod_none.triggered.connect(self.set_normal_mode)
        self.demod_group.addAction(self.action_demod_none)
        self.demod_menu.addAction(self.action_demod_none)

        self.demod_menu.addSeparator()

        # --- SUBMENÚ: FM ---
        self.fm_menu = QMenu("FM", self)
        self.fm_menu.setStyleSheet("""
            QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
            QMenu::item:selected { background-color: #555555; }
        """)

        # Opciones dentro del submenú FM
        self.action_wbfm = QAction("WBFM (Radio Comercial)", self)
        self.action_wbfm.setCheckable(True)
        self.action_wbfm.triggered.connect(self.set_wbfm_mode)
        self.demod_group.addAction(self.action_wbfm)
        self.fm_menu.addAction(self.action_wbfm)

        self.action_wbfm_audio = QAction("WBFM (Audio en Vivo)", self)
        self.action_wbfm_audio.setCheckable(True)
        self.action_wbfm_audio.triggered.connect(self.set_wbfm_audio_mode) # Usará una función nueva
        self.demod_group.addAction(self.action_wbfm_audio)
        self.fm_menu.addAction(self.action_wbfm_audio)

        self.action_nbfm = QAction("Custom FM", self)
        self.action_nbfm.setCheckable(True)
        # self.action_nbfm.triggered.connect(...)
        self.demod_group.addAction(self.action_nbfm)
        self.fm_menu.addAction(self.action_nbfm)

        # Agregamos el submenú FM al menú principal de Demodulación
        self.demod_menu.addMenu(self.fm_menu)

        # Asignar menú al botón y agregar a la barra principal
        self.demod_btn.setMenu(self.demod_menu)
        self.toolbar.addWidget(self.demod_btn)

        # --- SUBMENÚ: DIGITAL ---
        self.digital_menu = QMenu("Digitales", self)
        self.digital_menu.setStyleSheet("""
            QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
            QMenu::item:selected { background-color: #555555; }
        """)

        self.action_wifi_ag = QAction("WiFi 802.11a/g (OFDM)", self)
        self.action_wifi_ag.setCheckable(True)
        self.action_wifi_ag.triggered.connect(self.set_wifi_ag_mode)
        
        self.demod_group.addAction(self.action_wifi_ag)
        self.digital_menu.addAction(self.action_wifi_ag)

        # Agregamos el submenú Digital al menú principal de Demodulación
        self.demod_menu.addMenu(self.digital_menu)

        # --- LADO DERECHO: CONTROLES ---
        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        form_layout = QFormLayout()

       # 1. FRECUENCIA CENTRAL (Común a todos)
        freq_layout = QHBoxLayout() # Layout horizontal para juntar el número y la unidad

        self.freq_input = QDoubleSpinBox()
        self.freq_input.setKeyboardTracking(False)
        self.freq_input.setDecimals(6) # Le damos bastantes decimales para que aguante conversiones
        self.freq_input.setRange(0.0, 6000000000.0) # Rango gigante para cubrir desde Hz a GHz
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Hz", "kHz", "MHz", "GHz"])
        self.unit_combo.setCurrentText("MHz")
        
        # Agregamos los dos elementos al layout horizontal
        freq_layout.addWidget(self.freq_input)
        freq_layout.addWidget(self.unit_combo)

        # Agregamos el layout compuesto al formulario
        form_layout.addRow(QLabel("FREQ CENTRAL:"), freq_layout)

        # Variables de estado para las unidades
        self.current_freq_multiplier = 1e6
        self.freq_input.setValue(state['center_freq'] / self.current_freq_multiplier)
        self.update_spinbox_step() # Ajusta el salto de las flechitas

        # Conexiones
        self.freq_input.valueChanged.connect(self.on_freq_changed)
        self.unit_combo.currentTextChanged.connect(self.on_unit_changed)

        # 2. SAMPLE RATE (Común, pero con opciones distintas según SDR)
        self.sr_label = QLabel("SAMP RATE:")
        self.sr_combo = QComboBox()
        if "HackRF One" in self.radio.nombre:
            self.sr_combo.addItems(["2 MHz", "4 MHz", "8 MHz", "10 MHz", "12.5 MHz", "16 MHz", "20 MHz"])
            self.sr_combo.setCurrentText("10 MHz")
        elif "RTL-SDR" in self.radio.nombre:
            self.sr_combo.addItems(["1.024 MHz", "2.048 MHz", "2.4 MHz", "2.88 MHz"])
            self.sr_combo.setCurrentText("2.4 MHz")
        elif "Nuand bladeRF x40" in self.radio.nombre:
            self.sr_combo.addItems(["2 MHz", "5 MHz", "10 MHz", "20 MHz", "28 MHz", "40 MHz"])
            self.sr_combo.setCurrentText("20 MHz")
        elif "Ettus USRP B200" in self.radio.nombre:
            # La B200 soporta casi cualquier rate (hasta 56 MHz), ponemos valores enteros seguros
            self.sr_combo.addItems(["2 MHz", "4 MHz", "8 MHz", "10 MHz", "16 MHz", "20 MHz", "32 MHz"])
            # Arrancamos en 2 MHz para evitar el Overflow apenas abre el programa
            self.sr_combo.setCurrentText("2 MHz") 
            
        self.sr_combo.currentTextChanged.connect(self.on_sr_changed)
        form_layout.addRow(self.sr_label, self.sr_combo)

        # 3. GANANCIAS (Aparecen, cambian de nombre o desaparecen)
        self.lna_combo = QComboBox() 
        self.vga_combo = QComboBox()

        if "HackRF One" in self.radio.nombre:
            self.lna_combo.addItems([f"{g} dB" for g in range(0, 48, 8)])
            self.lna_combo.setCurrentText("8 dB")
            self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
            form_layout.addRow(QLabel("LNA GAIN:"), self.lna_combo)

            self.vga_combo.addItems([f"{g} dB" for g in range(0, 64, 2)])
            self.vga_combo.setCurrentText("16 dB")
            self.vga_combo.currentTextChanged.connect(self.on_vga_changed)
            form_layout.addRow(QLabel("VGA GAIN:"), self.vga_combo)

        elif "Nuand bladeRF x40" in self.radio.nombre:
            self.lna_combo.addItems([f"{g} dB" for g in range(0, 61, 5)])
            self.lna_combo.setCurrentText("0 dB") 
            self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
            form_layout.addRow(QLabel("GLOBAL GAIN:"), self.lna_combo)
            
        elif "Ettus USRP B200" in self.radio.nombre:
            # La B200 tiene una ganancia unificada que va de 0 a ~73/76 dB. 
            self.lna_combo.addItems([f"{g} dB" for g in range(0, 76, 5)])
            self.lna_combo.setCurrentText("40 dB") # Arrancamos por la mitad
            self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
            form_layout.addRow(QLabel("RX GAIN:"), self.lna_combo)
            
        # Si es "rtlsdr", directamente NO agregamos los botones de ganancia al layout.

        # 4. FFT y TRACE (Común a todos)
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["512", "1024", "2048", "4096", "8192"])
        self.fft_combo.setCurrentText("4096")
        self.fft_combo.currentTextChanged.connect(self.on_fft_changed)
        form_layout.addRow(QLabel("TAMAÑO FFT:"), self.fft_combo)

        self.trace_combo = QComboBox()
        self.trace_combo.addItems(["White clear", "Max Hold", "Average"])
        self.trace_combo.setCurrentText("White clear")
        self.trace_combo.currentTextChanged.connect(self.trace_manager.set_mode)
        form_layout.addRow(QLabel("TRACE:"), self.trace_combo)

        # ---  BOTÓN ZERO SPAN ---
        self.zero_span_btn = QPushButton("Spam Cero")
        self.zero_span_btn.setCheckable(True)
        self.zero_span_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.zero_span_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #555;")
        self.zero_span_btn.clicked.connect(self.toggle_zero_span)
        self.zero_span_label = QLabel("MODO SA:")
        form_layout.addRow(self.zero_span_label, self.zero_span_btn)

        self.waterfall_checkbox = QCheckBox("Activar Waterfall")
        self.waterfall_checkbox.setStyleSheet("color: white; font-weight: bold;")
        self.waterfall_checkbox.stateChanged.connect(self.on_waterfall_toggled)
        self.waterfall_label = QLabel("ESPECTROGRAMA:")
        form_layout.addRow(self.waterfall_label, self.waterfall_checkbox)

        controls_layout.addLayout(form_layout)

        # 5. BOTONES DE AUDIO ESTÉREO
        self.audio_container = QWidget() # Creamos un contenedor
        audio_layout = QHBoxLayout(self.audio_container)
        audio_layout.setContentsMargins(0, 15, 0, 0)
        
        self.audio_l_btn = QPushButton("🔊 Canal L")
        self.audio_l_btn.setCheckable(True)
        self.audio_l_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.audio_l_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")
        self.audio_l_btn.clicked.connect(self.audio_manager.toggle_audio)
        
        self.audio_r_btn = QPushButton("🔊 Canal R")
        self.audio_r_btn.setCheckable(True)
        self.audio_r_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.audio_r_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")
        self.audio_r_btn.clicked.connect(self.audio_manager.toggle_audio)
        
        audio_layout.addWidget(self.audio_l_btn)
        audio_layout.addWidget(self.audio_r_btn)
        
        # Agregamos el contenedor al layout principal de controles
        controls_layout.addWidget(self.audio_container)
        
        # Ocultamos el contenedor por defecto al iniciar la app
        self.audio_container.hide() 

        # --- MÉTRICAS FM EN EL PANEL DERECHO ---
        self.fm_metrics_label = QLabel("")
        self.fm_metrics_label.setTextFormat(Qt.TextFormat.RichText)
        self.fm_metrics_label.setStyleSheet("background-color: #1e1e1e; padding: 10px; border-radius: 4px; border: 1px solid #444; margin-top: 15px;")
        self.fm_metrics_label.hide() # Lo ocultamos por defecto
        controls_layout.addWidget(self.fm_metrics_label)

        # --- MÉTRICAS ESTÉREO (SEPARACIÓN) ---
        self.stereo_metrics_label = QLabel("")
        self.stereo_metrics_label.setTextFormat(Qt.TextFormat.RichText)
        self.stereo_metrics_label.setStyleSheet("background-color: #1e1e1e; padding: 10px; border-radius: 4px; border: 1px solid #444; margin-top: 10px;")
        self.stereo_metrics_label.hide()
        controls_layout.addWidget(self.stereo_metrics_label)

        # --- MÉTRICAS WIFI (SIGNAL) ---
        self.wifi_metrics_label = QLabel("")
        self.wifi_metrics_label.setTextFormat(Qt.TextFormat.RichText)
        self.wifi_metrics_label.setStyleSheet("background-color: #1e1e1e; padding: 10px; border-radius: 4px; border: 1px solid #444; margin-top: 10px;")
        self.wifi_metrics_label.hide()
        controls_layout.addWidget(self.wifi_metrics_label)

        # --- MÉTRICAS WIFI HW (CFO/SNR) ---
        self.wifi_hw_metrics_label = QLabel("")
        self.wifi_hw_metrics_label.setTextFormat(Qt.TextFormat.RichText)
        self.wifi_hw_metrics_label.setStyleSheet("background-color: #1e1e1e; padding: 10px; border-radius: 4px; border: 1px solid #444; margin-top: 10px;")
        self.wifi_hw_metrics_label.hide()
        controls_layout.addWidget(self.wifi_hw_metrics_label)
        
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedWidth(300)
        main_layout.addWidget(controls_widget, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

   
class StartupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEMODULADOR SDR")
        self.resize(QSize(400, 350))
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        layout = QVBoxLayout()
        label = QLabel("Dispositivos SDR detectados por USB:")
        label.setStyleSheet("font-size: 17px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(label)

        self.scan_btn = QPushButton("🔄 Escanear Puertos USB")
        self.scan_btn.setCursor(Qt.CursorShape.PointingHandCursor) # Cambia la flecha por la manito
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444; 
                color: white; 
                font-weight: bold; 
                padding: 10px 15px; 
                border-radius: 6px; 
                border: 1px solid #5a5a5a;
            }
            QPushButton:hover {
                background-color: #555555;
                border: 1px solid #777777;
            }
            QPushButton:pressed {
                background-color: #2b2b2b;
                border: 1px solid #222222;
            }
        """)
        self.scan_btn.clicked.connect(self.scan_devices)
        layout.addWidget(self.scan_btn)

        self.device_list = QListWidget()
        self.device_list.setStyleSheet("""
            QListWidget { background-color: #1e1e1e; border: 1px solid #444; font-size: 14px; outline: none; }
            QListWidget::item { padding: 15px; }
            QListWidget::item:selected { background-color: #0077FF; color: white; }
            QListWidget::item:hover { background-color: #444444; }
        """)
        self.device_list.itemClicked.connect(self.launch_main_window)
        layout.addWidget(self.device_list)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.scan_devices()

    def scan_devices(self):
        self.device_list.clear()
        devices_found = 0

        try:
            if usb.core.find(idVendor=0x1d50, idProduct=0x6089):
                self.device_list.addItem("HackRF One")
                devices_found += 1
        except: pass
        try:
            if usb.core.find(idVendor=0x0bda, idProduct=0x2838):
                self.device_list.addItem("RTL-SDR")
                devices_found += 1
        except: pass
        try:
            if usb.core.find(idVendor=0x2cf0) or usb.core.find(idVendor=0x1d50, idProduct=0x6066):
                self.device_list.addItem("Nuand bladeRF x40")
                devices_found += 1
        except: pass
        try:
            if usb.core.find(idVendor=0x2500):
                self.device_list.addItem("Ettus USRP B200")
                devices_found += 1
        except: pass

        if devices_found == 0:
            item = QListWidgetItem("⚠️ No se encontraron dispositivos SDR")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.device_list.addItem(item)

    def launch_main_window(self, item):
        device_name = item.text()
        radio_handler = None
        
        # --- INSTANCIACIÓN MODULAR DEL HARDWARE ---
        if "HackRF" in device_name:
            print("Iniciando HackRF...")
            radio_handler = HackRFHandler(rx_callback=None)
        elif "RTL-SDR" in device_name:
            print("Iniciando RTL-SDR...")
            radio_handler = RtlSdrHandler(rx_callback=None)
        elif "bladeRF" in device_name:
            print("Iniciando bladeRF...")
            radio_handler = BladeRFHandler(rx_callback=None)
        elif "Ettus USRP B200" in device_name:
            print("Iniciando Ettus B200...")
            radio_handler = USRPB200Handler(rx_callback=None)
        else:
            return

        # Pasamos el handler a la ventana principal
        self.main_app_window = MainWindow(radio_handler)
        self.main_app_window.show()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    startup_window = StartupWindow()
    startup_window.show()
    sys.exit(app.exec())