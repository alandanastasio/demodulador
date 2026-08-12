import sys
import os
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import datetime
import usb.core
from PyQt6.QtCore import QSize, Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QPainterPath, QIcon, QPainter
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                           QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QFormLayout, 
                             QToolBar, QToolButton, QMenu, QFileDialog, QListWidget,
                             QPushButton, QListWidgetItem, QGridLayout, QCheckBox, QMessageBox)
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
from dsp.demoduladores.lte import DemoduladorLTE
# Managers
from marker_manager import MarkerManager
from playback_manager import PlaybackManager
from ui_builder import build_ui
from plot_renderer import render_plot
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
        icon_path = os.path.join(os.path.dirname(__file__), "logo.demod.png")
        self.setWindowIcon(QIcon(icon_path))
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

        # Construimos la interfaz gráfica delegada
        build_ui(self, state)
        self.set_normal_mode()

        # Conectamos el actualizador de gráficos delegando a render_plot
        emitter.data_updated.connect(lambda *args: render_plot(self, state, *args))

        # --- Creamos el eje X matemático en memoria ANTES de encender el SDR ---
        self.update_x_axis()
        
        # 3. ENCENDIDO DE LA RADIO
        self.radio.configurar(state['sample_rate'], state['center_freq'])
        self.radio.start_rx()

    def procesar_muestras_iq(self, c_samples):
        # 1. Grabación de muestras I/Q crudas (si el usuario activó la grabación)
        if state['is_recording'] and c_samples is not None:
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
        if hasattr(self, 'lte_q1_stack') and self.lte_q1_stack.indexOf(self.freq_plot) != -1:
            self.lte_q1_stack.removeWidget(self.freq_plot)
            from PyQt6.QtWidgets import QWidget
            self.lte_q1_stack.insertWidget(0, QWidget())
            
        self.layout_wbfm.addWidget(self.freq_plot, 0, 0)
        self.layout_wbfm.setRowStretch(0, 1)
        self.layout_wbfm.setRowStretch(1, 1)
        self.layout_wbfm.setColumnStretch(0, 1)
        self.layout_wbfm.setColumnStretch(1, 1)
        if hasattr(self, 'waterfall_checkbox'): 
            self.waterfall_checkbox.hide()
            self.waterfall_label.hide()
        if hasattr(self, 'waterfall_controls_widget'):
            self.waterfall_controls_widget.hide()
        if hasattr(self, 'wf_bottom_widget'):
            self.wf_bottom_widget.hide()
        if hasattr(self, 'waterfall_line2'):
            self.waterfall_line2.hide()
        if hasattr(self, 'zero_span_btn'): 
            self.zero_span_btn.setChecked(False)
            state['zero_span'] = False
            self.zero_span_btn.hide()
            self.zero_span_label.hide()
        
        self.trace_manager.reset()
        
        self.modes_stack.setCurrentIndex(1) # PÁGINA WBFM
        self.audio_container.hide()
        self.fm_metrics_label.show() 
        self.stereo_metrics_label.show()
        self.wifi_metrics_label.hide()
        self.wifi_hw_metrics_label.hide()
        if hasattr(self, 'lte_metrics_label'):
            self.lte_metrics_label.hide()
        self.sr_combo.blockSignals(True)
        if self.sr_combo.findText("3.0 MHz (Decimado a 300k)") == -1:
            self.sr_combo.addItem("3.0 MHz (Decimado a 300k)")
        self.sr_combo.setCurrentText("3.0 MHz (Decimado a 300k)")
        self.sr_combo.setEnabled(False) 
        self.sr_combo.blockSignals(False)
        
        self.fft_combo.blockSignals(True)
        if hasattr(self, 'sa_fft_size_text'):
            self.fft_combo.setCurrentText(self.sa_fft_size_text)
            state['fft_size'] = int(self.sa_fft_size_text)
        self.fft_combo.setEnabled(True)
        self.fft_combo.blockSignals(False)

        state['demod_mode'] = 'wbfm'
        state['sample_rate'] = 3.0e6 
        
        self.freq_plot.show()
        self.demodulador_actual = DemoduladorWBFM()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
        self.radio.set_sample_rate(state['sample_rate'])
        self.freq_input.setValue(100.0)
        self.update_x_axis()
        self._restore_panels()

    def set_wifi_ag_mode(self):
        if hasattr(self, 'lte_q1_stack') and self.lte_q1_stack.indexOf(self.freq_plot) != -1:
            self.lte_q1_stack.removeWidget(self.freq_plot)
            from PyQt6.QtWidgets import QWidget
            self.lte_q1_stack.insertWidget(0, QWidget())
            
        self.layout_wifi.addWidget(self.freq_plot, 0, 0)
        self.layout_wifi.setRowStretch(0, 1)
        self.layout_wifi.setRowStretch(1, 1)
        self.layout_wifi.setRowStretch(2, 1)
        self.layout_wifi.setColumnStretch(0, 1)
        self.layout_wifi.setColumnStretch(1, 1)
        if hasattr(self, 'waterfall_checkbox'): 
            self.waterfall_checkbox.hide()
            self.waterfall_label.hide()
        if hasattr(self, 'waterfall_controls_widget'):
            self.waterfall_controls_widget.hide()
        if hasattr(self, 'wf_bottom_widget'):
            self.wf_bottom_widget.hide()
        if hasattr(self, 'waterfall_line2'):
            self.waterfall_line2.hide()
        if hasattr(self, 'zero_span_btn'): 
            self.zero_span_btn.setChecked(False)
            state['zero_span'] = False
            self.zero_span_btn.hide()
            self.zero_span_label.hide()
        
        self.trace_manager.reset()
        
        self.modes_stack.setCurrentIndex(2) # PÁGINA WIFI
        self.audio_container.hide()
        self.fm_metrics_label.hide()
        self.stereo_metrics_label.hide()
        self.wifi_metrics_label.show()
        self.wifi_hw_metrics_label.show()
        if hasattr(self, 'lte_metrics_label'):
            self.lte_metrics_label.hide()

        state['demod_mode'] = 'wifi_ag'
        state['sample_rate'] = 20e6 
        
        if hasattr(self.radio, 'set_muestras_por_bloque'):
            muestras = int(state['sample_rate'] * 0.002)
            pot2 = 1
            while pot2 < muestras: pot2 *= 2
            self.radio.set_muestras_por_bloque(pot2)

        self.demodulador_actual = DemoduladorWiFiAG()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
        self.radio.set_sample_rate(state['sample_rate'])
        
        self.unit_combo.setCurrentText("GHz")
        self.freq_input.setValue(2.412)

        if state.get('demod_mode', 'none') == 'none':
            self.sa_sample_rate_text = self.sr_combo.currentText()
            
        self.sr_combo.blockSignals(True)
        if self.sr_combo.findText("20 MHz") != -1:
            self.sr_combo.setCurrentText("20 MHz")
        elif self.sr_combo.findText("20.0 MHz") != -1:
            self.sr_combo.setCurrentText("20.0 MHz")
        self.sr_combo.setEnabled(False)
        self.sr_combo.blockSignals(False)
        
        self.fft_combo.blockSignals(True)
        if hasattr(self, 'sa_fft_size_text'):
            self.fft_combo.setCurrentText(self.sa_fft_size_text)
            state['fft_size'] = int(self.sa_fft_size_text)
        self.fft_combo.setEnabled(True)
        self.fft_combo.blockSignals(False)
        self.freq_plot.show()
        self._restore_panels()

    def set_lte_mode(self, bw_mhz=5):
        # Mapeo de ancho de banda LTE a frecuencia de muestreo y FFT
        bw_to_config = {
            1.4: (1.92e6, 128),
            3:   (3.84e6, 256),
            5:   (7.68e6, 512),
            10:  (15.36e6, 1024),
            15:  (23.04e6, 1536),
            20:  (30.72e6, 2048),
        }
        sample_rate, fft_size = bw_to_config.get(bw_mhz, (7.68e6, 512))
        
        # Insertamos el espectro en el stack del cuadrante 1 (reemplazando el widget temporal si existe)
        current_w = self.lte_q1_stack.widget(0)
        if current_w != self.freq_plot:
            self.lte_q1_stack.removeWidget(current_w)
            self.lte_q1_stack.insertWidget(0, self.freq_plot)
        
        # Mantenemos el índice según lo que esté seleccionado en el menú
        self.lte_q1_stack.setCurrentIndex(0 if self.action_q1_espectro.isChecked() else 1)
        
        self.layout_lte.setRowStretch(0, 1)
        self.layout_lte.setRowStretch(1, 1)
        self.layout_lte.setRowStretch(2, 1)
        self.layout_lte.setColumnStretch(0, 1)
        self.layout_lte.setColumnStretch(1, 1)
        if hasattr(self, 'waterfall_checkbox'): 
            self.waterfall_checkbox.hide()
            self.waterfall_label.hide()
        if hasattr(self, 'waterfall_controls_widget'):
            self.waterfall_controls_widget.hide()
        if hasattr(self, 'wf_bottom_widget'):
            self.wf_bottom_widget.hide()
        if hasattr(self, 'waterfall_line2'):
            self.waterfall_line2.hide()
        if hasattr(self, 'zero_span_btn'): 
            self.zero_span_btn.setChecked(False)
            state['zero_span'] = False
            self.zero_span_btn.hide()
            self.zero_span_label.hide()
        
        self.trace_manager.reset()
        
        self.modes_stack.setCurrentIndex(3) # PÁGINA LTE
        self.audio_container.hide()
        self.fm_metrics_label.hide()
        self.stereo_metrics_label.hide()
        self.wifi_metrics_label.hide()
        self.wifi_hw_metrics_label.hide()
        self.lte_metrics_label.show()

        # Guardamos el state del SA antes de pisar todo
        if state.get('demod_mode', 'none') == 'none':
            self.sa_sample_rate_text = self.sr_combo.currentText()
            self.sa_fft_size_text = self.fft_combo.currentText()

        state['demod_mode'] = 'lte'
        state['sample_rate'] = sample_rate
        state['fft_size'] = fft_size
        
        if hasattr(self.radio, 'set_muestras_por_bloque'):
            muestras = int(sample_rate * 0.002)
            pot2 = 1
            while pot2 < muestras: pot2 *= 2
            self.radio.set_muestras_por_bloque(pot2)

        self.demodulador_actual = DemoduladorLTE()
        self.demodulador_actual.configurar(sample_rate, fft_size)
        self.radio.set_sample_rate(sample_rate)
        
        self.unit_combo.setCurrentText("MHz")
        self.freq_input.setValue(2132.5)

        # Fijar y deshabilitar los combos de SR y FFT
        sr_text = f"{sample_rate / 1e6:.2f} MHz".replace(".00 ", " ")
        self.sr_combo.blockSignals(True)
        # Agregar el valor exacto si no existe en la lista
        if self.sr_combo.findText(sr_text) == -1:
            self.sr_combo.addItem(sr_text)
        self.sr_combo.setCurrentText(sr_text)
        self.sr_combo.setEnabled(False)
        self.sr_combo.blockSignals(False)
        
        self.fft_combo.blockSignals(True)
        fft_text = str(fft_size)
        if self.fft_combo.findText(fft_text) == -1:
            self.fft_combo.addItem(fft_text)
        self.fft_combo.setCurrentText(fft_text)
        self.fft_combo.setEnabled(False)
        self.fft_combo.blockSignals(False)
        
        self.update_x_axis()
        self._restore_panels()

    def set_wbfm_audio_mode(self):
        self.set_wbfm_mode() 
        self.audio_container.show()
        
        state['demod_mode'] = 'wbfm_audio'
        
        # Cargamos el Plugin de Audio
        self.demodulador_actual = DemoduladorWBFMAudio()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])

    def set_normal_mode(self):
        if hasattr(self, 'lte_q1_stack') and self.lte_q1_stack.indexOf(self.freq_plot) != -1:
            self.lte_q1_stack.removeWidget(self.freq_plot)
            from PyQt6.QtWidgets import QWidget
            self.lte_q1_stack.insertWidget(0, QWidget())
            
        self.layout_normal.addWidget(self.freq_plot, 0, 0)
        # En modo normal, la cascada va abajo (si está activa)
        self.layout_normal.addWidget(self.waterfall_widget, 1, 0)
        self.layout_normal.setRowStretch(0, 1)
        if getattr(self, 'waterfall_enabled', False):
            self.waterfall_widget.show()
            self.layout_normal.setRowStretch(1, 1)
        else:
            self.waterfall_widget.hide()
            self.layout_normal.setRowStretch(1, 0)
        self.modes_stack.setCurrentIndex(0) # PÁGINA NORMAL
        self.radio.set_muestras_por_bloque(32768)
        if hasattr(self, 'waterfall_checkbox'): 
            self.waterfall_checkbox.show()
            self.waterfall_label.show()
        if hasattr(self, 'waterfall_controls_widget'):
            self.waterfall_controls_widget.show()
        if hasattr(self, 'wf_bottom_widget'):
            self.wf_bottom_widget.show()
        if hasattr(self, 'waterfall_line2'):
            self.waterfall_line2.show()
        if hasattr(self, 'zero_span_btn'): 
            self.zero_span_btn.show()
            self.zero_span_label.show()
            
        self.trace_manager.reset()
        
        self.audio_container.hide()
        self.fm_metrics_label.hide()
        self.stereo_metrics_label.hide()
        self.wifi_metrics_label.hide()
        self.wifi_hw_metrics_label.hide()
        if hasattr(self, 'lte_metrics_label'):
            self.lte_metrics_label.hide()
        
        if self.audio_l_btn.isChecked() or self.audio_r_btn.isChecked():
            self.audio_l_btn.setChecked(False)
            self.audio_r_btn.setChecked(False)
            self.audio_manager.toggle_audio()
        
        self.sr_combo.blockSignals(True)
        idx = self.sr_combo.findText("3.0 MHz (Decimado a 300k)")
        if idx != -1: self.sr_combo.removeItem(idx)
        
        if hasattr(self, 'sa_sample_rate_text'):
            self.sr_combo.setCurrentText(self.sa_sample_rate_text)
            
        self.sr_combo.setEnabled(True) 
        self.sr_combo.blockSignals(False)
        
        # Restaurar FFT combo si venimos de LTE
        self.fft_combo.blockSignals(True)
        if hasattr(self, 'sa_fft_size_text'):
            self.fft_combo.setCurrentText(self.sa_fft_size_text)
            state['fft_size'] = int(self.sa_fft_size_text)
        self.fft_combo.setEnabled(True)
        self.fft_combo.blockSignals(False)
        
        state['demod_mode'] = 'none'
        self.freq_plot.show()
        self.demodulador_actual = SpectrumAnalyzer()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
        self.on_sr_changed(self.sr_combo.currentText()) 
        self._restore_panels()
        self.update_x_axis()

    def on_freq_changed(self, val):
        state['center_freq'] = val * self.current_freq_multiplier
        self.radio.set_freq(state['center_freq'])
        self.trace_manager.reset()
        self.update_x_axis()

    def on_sr_changed(self, text):
        if not text: return
        
        if state.get('demod_mode', 'none') == 'none':
            self.sa_sample_rate_text = text
            
        val_mhz = float(text.replace(" MHz", ""))
        state['sample_rate'] = val_mhz * 1e6
        
        self.radio.set_sample_rate(state['sample_rate'])
        
        # Si hay un plugin activo, le avisamos que cambió el sample rate
        if self.demodulador_actual is not None:
            self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])
            
        self.trace_manager.reset()
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
        if not hasattr(self, 'all_panels'): return
        
        self._saved_visibility = {w: w.isVisible() for w in self.all_panels}
        
        layout = widget.parentWidget().layout() if widget.parentWidget() else None
        
        from PyQt6.QtWidgets import QGridLayout
        if isinstance(layout, QGridLayout):
            self._saved_row_stretches = {i: layout.rowStretch(i) for i in range(layout.rowCount())}
            self._saved_col_stretches = {i: layout.columnStretch(i) for i in range(layout.columnCount())}
            
            idx = layout.indexOf(widget)
            if idx != -1:
                row, col, rowSpan, colSpan = layout.getItemPosition(idx)
                for i in range(layout.rowCount()):
                    layout.setRowStretch(i, 1 if (row <= i < row + rowSpan) else 0)
                for i in range(layout.columnCount()):
                    layout.setColumnStretch(i, 1 if (col <= i < col + colSpan) else 0)
                    
        for w in self.all_panels:
            if w is not widget:
                w.hide()
                
        widget.show()
        self._maximized_widget = widget
        self._maximized_layout = layout if isinstance(layout, QGridLayout) else None

    def _restore_panels(self):
        if self._maximized_widget is None: return
        
        for w, was_visible in self._saved_visibility.items():
            w.setVisible(was_visible)
            
        if hasattr(self, '_maximized_layout') and self._maximized_layout:
            layout = self._maximized_layout
            for i, s in getattr(self, '_saved_row_stretches', {}).items():
                layout.setRowStretch(i, s)
            for i, s in getattr(self, '_saved_col_stretches', {}).items():
                layout.setColumnStretch(i, s)
                
        self._maximized_widget = None
        self._maximized_layout = None
        self._saved_visibility = {}
        self._saved_row_stretches = {}
        self._saved_col_stretches = {}
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
                self.waterfall_widget.setTitle("Espectrograma (Waterfall)")
                self.waterfall_widget.setLabel('left', 'Tiempo [s]')
                self.waterfall_widget.enableAutoRange(axis='xy')
                self.waterfall_widget.getViewBox().invertY(True)
                self.waterfall_widget.setXLink(self.freq_plot)
                self.waterfall_widget.show()
                self.layout_normal.setRowStretch(1, 1)
        else:
            if state.get('demod_mode') == 'none':
                self.waterfall_widget.setTitle("")
                self.waterfall_widget.setLabel('left', '')
                self.waterfall_widget.getViewBox().invertY(False)
                self.waterfall_widget.setXLink(None)
                # Limpiar el ImageItem
                self.waterfall_image.clear()
                self.waterfall_widget.hide()
                self.layout_normal.setRowStretch(1, 0)

    def change_waterfall_lines(self, delta):
        new_val = getattr(self, 'waterfall_lines', 200) + delta
        if new_val < 50: new_val = 50
        if new_val > 1000: new_val = 1000
        self.waterfall_lines = new_val
        if hasattr(self, 'wf_lines_label'):
            self.wf_lines_label.setText(f"{self.waterfall_lines} líneas")

    def on_smooth_toggled(self, state):
        smooth_on = (state == Qt.CheckState.Checked.value or state == 2)
        self.waterfall_widget.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth_on)
        self.waterfall_widget.update()

    def save_waterfall(self):
        if not getattr(self, 'waterfall_enabled', False) or self.waterfall_buffer is None:
            return
            
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Guardar Espectrograma", "", "Imágenes PNG (*.png)"
        )
        if not filepath:
            return
            
        if not filepath.lower().endswith('.png'):
            filepath += '.png'
            
        try:
            exporter = pg.exporters.ImageExporter(self.waterfall_widget.plotItem)
            exporter.export(filepath)
            QMessageBox.information(self, "Éxito", f"Espectrograma guardado correctamente en:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar la imagen:\n{str(e)}")
