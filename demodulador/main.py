import sys
import numpy as np
import pyqtgraph as pg
import datetime
import usb.core
import queue
import sounddevice as sd

from PyQt6.QtCore import QSize, Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QFormLayout, 
                             QToolBar, QToolButton, QMenu, QFileDialog, QListWidget,
                             QPushButton, QListWidgetItem, QGridLayout)

# --- IMPORTACIÓN DE NUESTROS MÓDULOS MODULARES ---
# Hardware
from hardware.hackrf_handler import HackRFHandler
from hardware.rtlsdr_handler import RtlSdrHandler
from hardware.nuand_bladerf_handler import BladeRFHandler
# DSP (Plugins)
from dsp.demoduladores.wbfm import DemoduladorWBFM
from dsp.demoduladores.wbfm_audio import DemoduladorWBFMAudio
from dsp.demoduladores.sa import SpectrumAnalyzer
# Managers
from marker_manager import MarkerManager
from playback_manager import PlaybackManager
from trace_manager import TraceManager

# --- ESTADO GLOBAL (Solo cosas de la UI y configuración general) ---
state = {
    'fft_size': 4096,
    'center_freq': 100e6,
    'sample_rate': 10e6,
    'demod_mode': 'none',
    'play_audio': False,
    'audio_queue': queue.Queue(maxsize=20),
    'is_recording': False,
    'recorded_samples': []
}

class SignalEmitter(QObject):
    # Definimos la señal para actualizar los gráficos desde otros hilos
    data_updated = pyqtSignal(np.ndarray, np.ndarray, object, object, object, object, object, object,object)

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

        # Iniciamos el manager de trazos
        self.trace_manager = TraceManager()

        # Inicializamos el gestor de grabación/reproducción
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
                
                # 3. Gestión de Audio: Enviamos a la cola de sounddevice solo si la escucha 
                # está activa y si el plugin actual realmente generó muestras de audio.
                if state.get('play_audio', False) and resultados.get('audio_out') is not None:
                    if not state['audio_queue'].full():
                        state['audio_queue'].put(resultados['audio_out'])
                
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
                    resultados.get('mpx_time')
                )

    # === MÉTODOS DE LA UI (BOTONES Y MENÚS) ===

    def set_wbfm_mode(self):
        self.audio_container.hide()
        # Configuramos las pantallas
        self.q2_widget.setTitle("Espectro MPX (Audio Demodulado)")
        self.q2_widget.setLabel('bottom', 'Frecuencia [kHz]')
        self.q2_widget.setLabel('left', 'Magnitud [dB]')
        self.q2_widget.setXRange(0, 100)
        self.q2_widget.setYRange(-80, 20)

        self.q3_widget.setTitle("Señal Demodulada en el Tiempo")
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
        
        self.q2_widget.show()
        self.q3_widget.show()
        self.q4_container.show() # Mostramos el contenedor entero
        self.plot_layout.setRowStretch(1, 1)

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

    def set_wbfm_audio_mode(self):
        self.set_wbfm_mode() 
        self.audio_container.show()
        
        state['demod_mode'] = 'wbfm_audio'
        
        # Cargamos el Plugin de Audio
        self.demodulador_actual = DemoduladorWBFMAudio()
        self.demodulador_actual.configurar(state['sample_rate'], state['fft_size'])

    def set_normal_mode(self):
        self.q2_widget.hide()
        self.q3_widget.hide()
        self.q4_container.hide() # Ocultamos el contenedor

        self.plot_layout.setRowStretch(1, 0)
        
        self.q2_widget.setTitle("1-2 (Vacío)")
        self.q2_curve.setData([], []) 
        self.q3_widget.setTitle("2-1 (Vacío)")
        self.q3_curve.setData([], [])
        
        # Vaciamos L y R y las metricas
        self.q4L_widget.setTitle("Canal Izquierdo (L) - Vacío")
        self.q4R_widget.setTitle("Canal Derecho (R) - Vacío")
        self.q4L_curve.setData([], [])
        self.q4R_curve.setData([], [])
        self.audio_container.hide()
        self.fm_metrics_label.hide()
        self.fm_metrics_label.setText("")
        self.stereo_metrics_label.hide()
        self.stereo_metrics_label.setText("")
        
        if self.audio_l_btn.isChecked() or self.audio_r_btn.isChecked():
            self.audio_l_btn.setChecked(False)
            self.audio_r_btn.setChecked(False)
            self.toggle_audio()
        
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
        state['play_audio'] = False
        self.radio.close()
        event.accept()
        
    def keyPressEvent(self, event):
        # Le pasamos el evento al manager. Si lo pudo procesar (porque era una flechita),
        # devuelve True. Si devolvió False, dejamos que la ventana haga lo suyo por defecto.
        handled = self.marker_manager.handle_key_press(
            event, 
            self.f_axis, 
            getattr(self, 'last_f_axis_audio', None)
        )
        if not handled:
            super().keyPressEvent(event)

    def toggle_pause(self):
        self.is_paused = self.pause_btn.isChecked()
        
        if self.is_paused:
            self.pause_btn.setText("▶")
            self.pause_btn.setStyleSheet("background-color: #488BD8; color: black; font-size: 16px; font-weight: bold; border-radius: 4px; margin: 4px;")
        else:
            self.pause_btn.setText("⏸")
            self.pause_btn.setStyleSheet("background-color: #444; color: white; font-size: 16px; font-weight: bold; border-radius: 4px; margin: 4px;")


    def toggle_audio(self):
        play_l = self.audio_l_btn.isChecked()
        play_r = self.audio_r_btn.isChecked()
        
        # Cambiamos los colores (Cyan para L, Magenta para R)
        self.audio_l_btn.setStyleSheet("background-color: #00FFFF; color: black; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #00CCCC;" if play_l else "background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")
        self.audio_r_btn.setStyleSheet("background-color: #FF00FF; color: black; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #CC00CC;" if play_r else "background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")

        state['play_audio'] = play_l or play_r
        state['play_audio_L'] = play_l
        state['play_audio_R'] = play_r

        # Si hay alguno encendido y el stream no está corriendo, lo iniciamos
        if state['play_audio'] and (self.audio_stream is None or not self.audio_stream.active):
            # Limpiamos buffers (Ahora es una matriz vacía de 2 columnas)
            state['audio_buffer'] = np.zeros((0, 2), dtype=np.float32)
            while not state['audio_queue'].empty():
                state['audio_queue'].get()

            def audio_callback(outdata, frames, time, status):
                try:
                    while len(state['audio_buffer']) < frames:
                        new_data = state['audio_queue'].get_nowait()
                        # Si recibimos mono por accidente, lo duplicamos a estéreo
                        if new_data.ndim == 1:
                            new_data = np.column_stack((new_data, new_data))
                        # Apilamos las filas
                        state['audio_buffer'] = np.vstack((state['audio_buffer'], new_data))
                except queue.Empty:
                    pass
                
                if len(state['audio_buffer']) >= frames:
                    chunk = state['audio_buffer'][:frames].copy()
                    state['audio_buffer'] = state['audio_buffer'][frames:]
                    
                    # --- ASIGNACIÓN DIRECTA Y EXPLÍCITA AL HARDWARE ---
                    if state.get('play_audio_L', False):
                        outdata[:, 0] = chunk[:, 0] # Escribir audio al parlante L
                    else:
                        outdata[:, 0] = 0.0         # Forzar silencio en L
                        
                    if state.get('play_audio_R', False):
                        outdata[:, 1] = chunk[:, 1] # Escribir audio al parlante R
                    else:
                        outdata[:, 1] = 0.0         # Forzar silencio en R
                        
                else:
                    outdata.fill(0.0)
                    state['audio_buffer'] = np.zeros((0, 2), dtype=np.float32)

            # Iniciamos el stream (channels=2 para que el OS sepa que es estéreo)
            self.audio_stream = sd.OutputStream(
                samplerate=48000, 
                channels=2, 
                dtype='float32',
                callback=audio_callback
            )
            self.audio_stream.start()
            
        # Si apagamos ambos botones y el stream sigue corriendo, lo detenemos
        elif not state['play_audio'] and self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

    def update_plot(self, PSD, raw_samples, PSD_audio=None, f_axis_audio=None, audio_L=None, audio_R=None, t_axis=None, fm_metrics=None, mpx_time=None):
        if self.is_paused:
            return
        
        if len(self.f_axis) == len(PSD):
            
            # --- LÓGICA DE TRACE ---
            display_psd = self.trace_manager.process(PSD)

            # Graficamos la señal procesada
            self.freq_plot_curve.setData(self.f_axis, display_psd)

            # --- Cacheamos los datos de MPX para que los markers no parpadeen ---
            if PSD_audio is not None:
                self.last_f_axis_audio = f_axis_audio
                self.last_PSD_audio = PSD_audio
            
            # --- LÓGICA DE MARKERS Y DELTAS ---
            self.marker_manager.update_render(
                display_psd, 
                self.f_axis, 
                PSD_audio, 
                getattr(self, 'last_f_axis_audio', None), 
                state.get('demod_mode')
            )
            

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

        # Agregamos a la grilla principal
        self.plot_layout.addWidget(self.freq_plot, 0, 0)
        self.plot_layout.addWidget(self.q2_widget, 0, 1) 
        self.plot_layout.addWidget(self.q3_widget, 1, 0) 
        self.plot_layout.addWidget(self.q4_container, 1, 1) 

        # Dejamos las curvas creadas
        self.q2_curve = self.q2_widget.plot([], pen=pg.mkPen(color="#C3FF00", width=1.5))
        self.q3_curve = self.q3_widget.plot([], pen=pg.mkPen(color="#FF9500", width=1.5))
        self.q4L_curve = self.q4L_widget.plot([], pen=pg.mkPen(color="#00FFFF", width=1.5))
        self.q4R_curve = self.q4R_widget.plot([], pen=pg.mkPen(color="#FF00FF", width=1.5))

        # Ocultamos por defecto
        self.q2_widget.hide()
        self.q3_widget.hide()
        self.q4_container.hide()

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

        # --- LADO DERECHO: CONTROLES ---
        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        form_layout = QFormLayout()

       # 1. FRECUENCIA CENTRAL (Común a todos)
        freq_layout = QHBoxLayout() # Layout horizontal para juntar el número y la unidad

        self.freq_input = QDoubleSpinBox()
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
            
        self.sr_combo.currentTextChanged.connect(self.on_sr_changed)
        form_layout.addRow(self.sr_label, self.sr_combo)

        # 3. GANANCIAS (Aparecen, cambian de nombre o desaparecen)
        self.lna_combo = QComboBox() # Creamos las variables para no romper callbacks
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
            self.lna_combo.setCurrentText("0 dB") # Arrancamos en 0 para no tener tanto ruido al inicio
            self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
            # Lo llamamos GLOBAL GAIN porque la bladeRF maneja una sola etapa unificada
            form_layout.addRow(QLabel("GLOBAL GAIN:"), self.lna_combo)
            
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

        controls_layout.addLayout(form_layout)

        # 5. BOTONES DE AUDIO ESTÉREO
        self.audio_container = QWidget() # Creamos un contenedor
        audio_layout = QHBoxLayout(self.audio_container)
        audio_layout.setContentsMargins(0, 15, 0, 0)
        
        self.audio_l_btn = QPushButton("🔊 Canal L")
        self.audio_l_btn.setCheckable(True)
        self.audio_l_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.audio_l_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")
        self.audio_l_btn.clicked.connect(self.toggle_audio)
        
        self.audio_r_btn = QPushButton("🔊 Canal R")
        self.audio_r_btn.setCheckable(True)
        self.audio_r_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.audio_r_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")
        self.audio_r_btn.clicked.connect(self.toggle_audio)
        
        audio_layout.addWidget(self.audio_l_btn)
        audio_layout.addWidget(self.audio_r_btn)
        
        # Agregamos el contenedor al layout principal de controles
        controls_layout.addWidget(self.audio_container)
        
        # Ocultamos el contenedor por defecto al iniciar la app
        self.audio_container.hide() 
        
        self.audio_stream = None

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