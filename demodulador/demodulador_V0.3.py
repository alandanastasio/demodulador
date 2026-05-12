### DEMODULADOR V0.3 ###

from PyQt6.QtCore import QSize, Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QFormLayout, 
                             QToolBar, QToolButton, QMenu, QFileDialog, QListWidget,
                             QPushButton, QListWidgetItem, QGridLayout)
import pyqtgraph as pg
import numpy as np
import signal
from python_hackrf import pyhackrf
import time
import datetime
import usb.core
import threading
from rtlsdr import RtlSdr
import bladerf
from scipy.signal import butter, lfilter


# Estado global para compartir entre la GUI y el hilo de C (callback)
state = {
    'fft_size': 4096,
    'center_freq': 100e6,
    'sample_rate': 10e6,
    'demod_mode': 'none'
}

# Buffer global para acumular las muestras de FM
fm_buffer = np.array([], dtype=np.complex128)

class SignalEmitter(QObject):
    data_updated = pyqtSignal(np.ndarray, np.ndarray, object, object)

emitter = SignalEmitter()

def process_iq_samples(c_samples):
    global fm_buffer
    
    if state['demod_mode'] == 'wbfm':
        fm_buffer = np.append(fm_buffer, c_samples)
        sr = int(state['sample_rate'])
        
        if len(fm_buffer) >= sr:
            chunk_1s = fm_buffer[:sr]
            fm_buffer = fm_buffer[sr:] # Guardamos el excedente para el próximo segundo
            
            # --- 1. ESPECTRO RF CRUDO (Para gráfico 1-1) ---
            # Hacemos la FFT directo de las muestras que entraron SIN filtrar
            raw_rf = chunk_1s - np.mean(chunk_1s)
            potencia = np.abs(np.fft.fftshift(np.fft.fft(raw_rf)))**2 / sr
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            
            # --- 2. FILTRADO (Solo para la demodulación) ---
            nyq = 0.5 * sr
            b, a = butter(4, 100e3 / nyq, btype='low')
            filtered_chunk = lfilter(b, a, chunk_1s)
            
            # --- 3. DEMODULACIÓN (AUDIO MPX para gráfico 1-2) ---
            # Discriminador polar para extraer frecuencia instantánea de la señal FILTRADA
            demod = np.angle(filtered_chunk[1:] * np.conj(filtered_chunk[:-1]))
            
            # FFT de la señal de audio
            demod = demod - np.mean(demod)
            fs_audio = len(demod)
            potencia_audio = np.abs(np.fft.fft(demod))**2 / fs_audio
            
            # Como la señal de audio es real, nos quedamos solo con la mitad positiva
            mitad = fs_audio // 2
            potencia_audio_pos = potencia_audio[:mitad]
            PSD_audio = 10.0 * np.log10(np.maximum(potencia_audio_pos, 1e-12))
            
            # Eje X del audio de 0 a 150 kHz
            f_axis_audio = np.linspace(0, (sr/2)/1e3, mitad)
            
            # Emitimos para graficar. Mandamos chunk_1s crudo en lugar de filtered_chunk 
            # para que si le das a "Grabar", se guarde el espectro real completo.
            emitter.data_updated.emit(PSD, chunk_1s, PSD_audio, f_axis_audio)
            
    else:
        # Lógica Normal
        fs = state['fft_size']
        if len(c_samples) >= fs:
            chunk = c_samples[:fs].copy()
            chunk = chunk - np.mean(chunk)
            potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk)))**2 / fs
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            centro = fs // 2
            PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0
            
            # En modo normal, los datos de audio van como None
            emitter.data_updated.emit(PSD, chunk, None, None)

def rx_callback(device, buffer, buffer_length, valid_length):
    accepted_samples = buffer[:valid_length].astype(np.int8)
    c_samples = (accepted_samples[0::2] + 1j * accepted_samples[1::2]) / 128.0

    # Mandamos las muestras crudas a la procesadora central
    process_iq_samples(c_samples)

    return 0

class MainWindow(QMainWindow):
    def __init__(self, sdr_device, device_type="hackrf"): 
        super().__init__()
        self.sdr = sdr_device
        self.device_type = device_type # Guardamos qué radio es

        if self.device_type == "bladerf":
            import bladerf
            self.rx_ch = self.sdr.Channel(bladerf.CHANNEL_RX(0))

        self.setWindowTitle("DEMODULADOR")
        self.resize(QSize(1200, 600))
        self.setMinimumSize(QSize(800, 400))

        # Color de fondo y texto de la ventana
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        # Variables para la grabación y reproducción
        self.is_recording = False
        self.is_paused = False
        self.recorded_samples = []
        self.playback_timer = QTimer()                            
        self.playback_timer.timeout.connect(self.playback_step)   
        self.playback_data = None                                 
        self.playback_index = 0
        self.is_looping = False    

        # --- VARIABLES DEL TRACE ---
        self.trace_mode = "White clear"
        self.max_hold_data = None
        self.avg_buffer = None
        self.avg_index = 0
        self.avg_count = 0
        self.AVG_MAX = 100                               

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
        self.record_action.triggered.connect(self.toggle_recording)
        
        self.play_action = QAction(" ▶ Reproducir Archivo", self)
        self.play_action.triggered.connect(lambda: self.load_and_play(loop=False))
        
        # Boton de reproducir en loop y de detener
        self.loop_action = QAction("🔁 Reproducir archivo en loop", self)
        self.loop_action.triggered.connect(lambda: self.load_and_play(loop=True))

        self.stop_play_action = QAction("⏹ Detener Reproducción", self)
        self.stop_play_action.triggered.connect(self.stop_playback)
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
        self.freq_plot_curve = self.freq_plot.plot([], pen=pg.mkPen(color='#FFD500', width=1.5))

        # --- SISTEMA DE MARKERS ---
        # Diccionario con la configuración y estado de cada marker/delta
        self.markers_info = {
            'M1': {'active': False, 'freq': state['center_freq']/1e6, 'color': '#00B000', 'item': pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush('#00B000'), symbol='d')},
            'D1': {'active': False, 'freq': state['center_freq']/1e6, 'color': "#00B000", 'item': pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush("#007E00"), symbol='t')}, # 't' es triángulo
            'M2': {'active': False, 'freq': state['center_freq']/1e6, 'color': "#0077FF", 'item': pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush("#0077FF"), symbol='d')},
            'D2': {'active': False, 'freq': state['center_freq']/1e6, 'color': "#0077FF", 'item': pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush("#0054C2"), symbol='t')}
        }
        self.current_moving_marker = None # Cuál se mueve al hacer clic
        
        self.marker_text_box = pg.TextItem(text="", color='#FFFFFF', fill=pg.mkBrush(0, 0, 0, 200), anchor=(1, 0))
        self.freq_plot.addItem(self.marker_text_box)
        self.marker_text_box.hide()

        self.freq_plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        self.freq_plot.setYRange(-70, 10)
        self.update_x_axis()

        # CONTENEDOR GRILLA (2x2)
        self.plot_container = QWidget()
        self.plot_layout = QGridLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(5) # Un pequeño margen entre los gráficos

        # Agregamos el espectro original a la grilla [Fila 0, Columna 0] (1-1)
        self.plot_layout.addWidget(self.freq_plot, 0, 0)

        # Creamos los cuadrantes genericos
        self.q2_widget = pg.PlotWidget(title="1-2 (Vacío)")
        self.q3_widget = pg.PlotWidget(title="2-1 (Vacío)")
        self.q4_widget = pg.PlotWidget(title="2-2 (Vacío)")

        # Agregamos los vacíos a sus respectivas posiciones
        self.plot_layout.addWidget(self.q2_widget, 0, 1) # [Fila 0, Columna 1] (1-2)
        self.plot_layout.addWidget(self.q3_widget, 1, 0) # [Fila 1, Columna 0] (2-1)
        self.plot_layout.addWidget(self.q4_widget, 1, 1) # [Fila 1, Columna 1] (2-2)

        # Dejamos la curva creada genéricamente
        self.q2_curve = self.q2_widget.plot([], pen=pg.mkPen(color='#00FF00', width=1.5))

        # Los ocultamos por defecto al iniciar el programa
        self.q2_widget.hide()
        self.q3_widget.hide()
        self.q4_widget.hide()

        # En vez de agregar solo freq_plot, agregamos el contenedor entero al layout principal
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
            action.triggered.connect(lambda checked, k=key: self.select_marker(k))
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
        self.action_none.triggered.connect(lambda: self.select_marker(None))
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
            action.triggered.connect(lambda checked, k=key: self.delete_marker(k))
            self.delete_menu.addAction(action)
            return action

        create_delete_action("❌ Eliminar M1", 'M1')
        create_delete_action("❌ Eliminar Delta 1", 'D1')
        create_delete_action("❌ Eliminar M2", 'M2')
        create_delete_action("❌ Eliminar Delta 2", 'D2')
        
        self.delete_menu.addSeparator()
        
        self.clear_markers_action = QAction("💥 Limpiar Todos", self)
        self.clear_markers_action.triggered.connect(self.clear_markers)
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
        if self.device_type == "hackrf":
            self.sr_combo.addItems(["2 MHz", "4 MHz", "8 MHz", "10 MHz", "12.5 MHz", "16 MHz", "20 MHz"])
            self.sr_combo.setCurrentText("10 MHz")
        elif self.device_type == "rtlsdr":
            self.sr_combo.addItems(["1.024 MHz", "2.048 MHz", "2.4 MHz", "2.88 MHz"])
            self.sr_combo.setCurrentText("2.4 MHz")
        elif self.device_type == "bladerf":
            self.sr_combo.addItems(["2 MHz", "5 MHz", "10 MHz", "20 MHz", "28 MHz", "40 MHz"])
            self.sr_combo.setCurrentText("20 MHz")
            
        self.sr_combo.currentTextChanged.connect(self.on_sr_changed)
        form_layout.addRow(self.sr_label, self.sr_combo)

        # 3. GANANCIAS (Aparecen, cambian de nombre o desaparecen)
        self.lna_combo = QComboBox() # Creamos las variables para no romper callbacks
        self.vga_combo = QComboBox()

        if self.device_type == "hackrf":
            self.lna_combo.addItems([f"{g} dB" for g in range(0, 48, 8)])
            self.lna_combo.setCurrentText("32 dB")
            self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
            form_layout.addRow(QLabel("LNA GAIN:"), self.lna_combo)

            self.vga_combo.addItems([f"{g} dB" for g in range(0, 64, 2)])
            self.vga_combo.setCurrentText("50 dB")
            self.vga_combo.currentTextChanged.connect(self.on_vga_changed)
            form_layout.addRow(QLabel("VGA GAIN:"), self.vga_combo)

        elif self.device_type == "bladerf":
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
        self.trace_combo.currentTextChanged.connect(self.on_trace_changed)
        form_layout.addRow(QLabel("TRACE:"), self.trace_combo)

        controls_layout.addLayout(form_layout)

        controls_layout.addLayout(form_layout)
        
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedWidth(300)
        main_layout.addWidget(controls_widget, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        emitter.data_updated.connect(self.update_plot)

    def on_trace_changed(self, text):
        self.trace_mode = text
        self.reset_traces()

    def reset_traces(self):
        self.max_hold_data = None
        self.avg_buffer = None
        self.avg_index = 0
        self.avg_count = 0

    def set_wbfm_mode(self):
        
        # --- CONFIGURACIÓN DE CUADRANTES PARA WBFM ---
        self.q2_widget.setTitle("Espectro MPX (Audio Demodulado)")
        self.q2_widget.setLabel('bottom', 'Frecuencia [kHz]')
        self.q2_widget.setLabel('left', 'Magnitud [dB]')
        self.q2_widget.setXRange(0, 100)
        self.q2_widget.setYRange(-80, 20)
        
        # Mostrar los cuadrantes
        self.q2_widget.show()
        self.q3_widget.show()
        self.q4_widget.show()

        # --- MANEJO DEL SAMPLE RATE VISUAL ---
        # Guardamos el valor que estaba seleccionado para restaurarlo después
        self.previous_sr_text = self.sr_combo.currentText()
        
        # Bloqueamos las señales para no mandar comandos accidentales a la SDR
        self.sr_combo.blockSignals(True)
        
        # Agregamos "300 kHz" si no existe, lo seleccionamos y lo desactivamos
        if self.sr_combo.findText("300 kHz") == -1:
            self.sr_combo.addItem("300 kHz")
        self.sr_combo.setCurrentText("300 kHz")
        self.sr_combo.setEnabled(False) 
        
        self.sr_combo.blockSignals(False)
        # --------------------------------------

        state['demod_mode'] = 'wbfm'
        state['sample_rate'] = 300e3 # Forzamos a 300 ksps
        
        # Aplicar el cambio a la SDR seleccionada
        if self.device_type == "hackrf":
            self.sdr.pyhackrf_stop_rx()
            self.sdr.pyhackrf_set_sample_rate(int(state['sample_rate']))
            self.sdr.pyhackrf_start_rx()
        elif self.device_type == "rtlsdr":
            self.sdr.sample_rate = state['sample_rate']
        elif self.device_type == "bladerf":
            self.rx_ch.sample_rate = int(state['sample_rate'])
            self.rx_ch.bandwidth = 200000

        self.update_x_axis() 
        
        # Sintonizamos el centro de FM (97.75 MHz)
        freq_fm_centro_hz = 97.75 * 1e6 
        display_val = freq_fm_centro_hz / self.current_freq_multiplier
        self.freq_input.setValue(display_val)

    def set_normal_mode(self):
        self.q2_widget.hide()
        self.q3_widget.hide()
        self.q4_widget.hide()

        # Limpiamos los títulos visualmente (por si otra función los vuelve a mostrar sin configurar)
        self.q2_widget.setTitle("1-2 (Vacío)")
        self.q2_curve.setData([], []) # Borra la línea verde de la pantalla
        
        # --- MANEJO DEL SAMPLE RATE VISUAL ---
        self.sr_combo.blockSignals(True)
        
        # Eliminamos el "300 kHz" temporal de la lista
        idx = self.sr_combo.findText("300 kHz")
        if idx != -1:
            self.sr_combo.removeItem(idx)
            
        # Restauramos el valor que el usuario tenía antes de entrar a WBFM
        if hasattr(self, 'previous_sr_text'):
            self.sr_combo.setCurrentText(self.previous_sr_text)
            
        # Volvemos a habilitar el control
        self.sr_combo.setEnabled(True) 
        self.sr_combo.blockSignals(False)
        # --------------------------------------
        
        state['demod_mode'] = 'none'
        
        # Leemos lo que dice el combo restaurado y aplicamos a la radio
        self.on_sr_changed(self.sr_combo.currentText()) 
        self.update_x_axis()

    def select_marker(self, key):
        self.current_moving_marker = key
        if key is not None:
            # Si el marker no estaba activo, lo prendemos y lo mostramos
            if not self.markers_info[key]['active']:
                self.markers_info[key]['active'] = True
                self.freq_plot.addItem(self.markers_info[key]['item'])
                self.marker_text_box.show()

    def clear_markers(self):
        for key, data in self.markers_info.items():
            if data['active']:
                self.freq_plot.removeItem(data['item'])
            data['active'] = False
        
        self.marker_text_box.hide()
        self.action_none.setChecked(True)
        self.current_moving_marker = None

    def delete_marker(self, key):
        # 1. Si el marker estaba activo, lo sacamos de la pantalla y lo marcamos apagado
        if self.markers_info[key]['active']:
            self.freq_plot.removeItem(self.markers_info[key]['item'])
            self.markers_info[key]['active'] = False
        
        # 2. Si justo estábamos moviendo el marker que acabamos de borrar, reseteamos la selección
        if self.current_moving_marker == key:
            self.action_none.setChecked(True)
            self.current_moving_marker = None
        
        # 3. Si ya no queda NINGÚN marker activo en pantalla, ocultamos el cuadro de texto negro
        if not any(m['active'] for m in self.markers_info.values()):
            self.marker_text_box.hide()

    def on_mouse_clicked(self, event):
        # Mueve únicamente el marker que está seleccionado en el menú
        if event.button() == Qt.MouseButton.LeftButton and self.current_moving_marker is not None:
            if self.freq_plot.sceneBoundingRect().contains(event.scenePos()):
                mouse_point = self.freq_plot.getViewBox().mapSceneToView(event.scenePos())
                self.markers_info[self.current_moving_marker]['freq'] = mouse_point.x()

    def toggle_pause(self):
        self.is_paused = self.pause_btn.isChecked()
        
        if self.is_paused:
            self.pause_btn.setText("▶")
            self.pause_btn.setStyleSheet("background-color: #488BD8; color: black; font-size: 16px; font-weight: bold; border-radius: 4px; margin: 4px;")
        else:
            self.pause_btn.setText("⏸")
            self.pause_btn.setStyleSheet("background-color: #444; color: white; font-size: 16px; font-weight: bold; border-radius: 4px; margin: 4px;")

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_samples = [] # <-- Limpiamos la lista correcta
            
            self.record_action.setText("⏹ Detener y Guardar")
            self.rec_play_btn.setStyleSheet("background-color: #8b0000; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
            print("Grabación de muestras IQ iniciada...")
            
            self.freq_input.setEnabled(False)
            self.sr_combo.setEnabled(False)
            self.fft_combo.setEnabled(False)
        else:
            self.is_recording = False
            
            # 1. Cambiar estado visual y FORZAR a la GUI a actualizarse
            self.record_action.setText("⏳ Guardando...")
            self.rec_play_btn.setText("⏳ Guardando...")
            self.rec_play_btn.setStyleSheet("background-color: #ff8c00; color: black; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
            QApplication.processEvents() # <- Esto actualiza la pantalla ANTES de que se congele

            if len(self.recorded_samples) > 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"muestras_iq_{timestamp}.npz"
                
                # Unir todos los fragmentos en un único array gigante 1D
                todas_las_muestras = np.concatenate(self.recorded_samples)
                
                # 2. Guardar SIN COMPRESIÓN. Es muchísimo más rápido para ruido de RF.
                np.savez(
                    filename,
                    raw_iq=todas_las_muestras,
                    center_freq=state['center_freq'],
                    sample_rate=state['sample_rate']
                )
                print(f"Grabación guardada exitosamente en: {filename}")
                print(f"Muestras totales grabadas: {len(todas_las_muestras)}")
                self.recorded_samples = []

            # 3. Restaurar apariencia original de los botones y controles
            self.record_action.setText("🔴 Iniciar Grabación")
            self.rec_play_btn.setText("Rec/Play")
            self.rec_play_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
            
            self.freq_input.setEnabled(True)
            self.sr_combo.setEnabled(True)
            self.fft_combo.setEnabled(True)

    def load_and_play(self, loop=False): # <-- Agregamos el flag
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar Grabación IQ", "", "Numpy Archives (*.npz)")
        if not filename:
            return

        self.sdr.pyhackrf_stop_rx()
        print(f"Cargando archivo: {filename}...")
        QApplication.processEvents()

        try:
            data = np.load(filename)
            self.playback_data = data['raw_iq']
            cf = data['center_freq']
            sr = data['sample_rate']
        except Exception as e:
            print(f"Error al leer el archivo: {e}")
            self.sdr.pyhackrf_start_rx()
            return

        self.is_looping = loop # <-- Guardamos el flag en la clase
        state['center_freq'] = float(cf)
        state['sample_rate'] = float(sr)
        self.freq_input.setValue(cf / 1e6)
        self.update_x_axis()

        # Bloquear y desbloquear controles
        self.freq_input.setEnabled(False)
        self.sr_combo.setEnabled(False)
        self.fft_combo.setEnabled(False)
        self.record_action.setEnabled(False)
        self.play_action.setEnabled(False)
        self.loop_action.setEnabled(False)        # Bloquear Play en Loop
        self.stop_play_action.setEnabled(True)    # Habilitar botón de Parar
        
        # Cambiar el texto del botón principal según el modo
        if self.is_looping:
            self.rec_play_btn.setText("🔁 Reproduciendo Loop...")
        else:
            self.rec_play_btn.setText("▶ Reproduciendo...")
            
        self.rec_play_btn.setStyleSheet("background-color: #004d99; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
        self.freq_plot_curve.setPen(pg.mkPen(color="#FF8C00", width=1.5))

        self.playback_index = 0
        self.playback_timer.start(33) 
        print("Reproducción iniciada.")

    def playback_step(self):
        fs = state['fft_size']
        
        # Calcular cuántas muestras saltar para simular la velocidad real
        # Avance = Sample Rate (muestras/seg) * tiempo del frame (0.033 seg)
        avance = int(state['sample_rate'] * 0.033) 

        if self.playback_index + fs > len(self.playback_data):
            if self.is_looping:
                # Si estamos en loop, simplemente rebobinamos al índice 0
                self.playback_index = 0 
            else:
                # Si es reproducción normal, detenemos todo
                self.stop_playback() 
                return

        # Agarrar el pedacito de muestras crudas correspondiente a este frame
        chunk = self.playback_data[self.playback_index : self.playback_index + fs].copy()
        
        # Avanzar el puntero en el tiempo
        self.playback_index += avance

        # Hacer la misma matemática de la FFT que hace el rx_callback
        chunk = chunk - np.mean(chunk)
        potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk)))**2 / fs
        PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
        centro = fs // 2
        PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0

        # Enviar los datos al gráfico
        emitter.data_updated.emit(PSD, chunk)
    
    def stop_playback(self):
        self.playback_timer.stop()
        self.is_looping = False
        
        # Rehabilitar todo
        self.freq_input.setEnabled(True)
        self.sr_combo.setEnabled(True)
        self.fft_combo.setEnabled(True)
        self.record_action.setEnabled(True)
        self.play_action.setEnabled(True)
        self.loop_action.setEnabled(True)
        self.stop_play_action.setEnabled(False)
        
        # Restaurar botón principal y color del gráfico
        self.rec_play_btn.setText("Rec/Play")
        self.rec_play_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
        self.freq_plot_curve.setPen(pg.mkPen(color='#FFD500', width=1.5))
        
        # Volver a encender la HackRF en vivo
        print("Reproducción finalizada o detenida. Volviendo a la antena.")
        self.sdr.pyhackrf_start_rx()

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

    def on_freq_changed(self, val):
        # Ahora multiplicamos el valor de la cajita por la unidad seleccionada
        state['center_freq'] = val * self.current_freq_multiplier
        
        if self.device_type == "hackrf":
            self.sdr.pyhackrf_set_freq(int(state['center_freq']))
        elif self.device_type == "rtlsdr":
            self.sdr.center_freq = state['center_freq']
        elif self.device_type == "bladerf":
            self.rx_ch.frequency = int(state['center_freq'])
            
        self.update_x_axis()

    def on_sr_changed(self, text):
        if not text: return #ignoramos señales vacias

        val_mhz = float(text.replace(" MHz", ""))
        state['sample_rate'] = val_mhz * 1e6
        
        if self.device_type == "hackrf":
            self.sdr.pyhackrf_stop_rx()
            self.sdr.pyhackrf_set_sample_rate(int(state['sample_rate']))
            bw = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(state['sample_rate'] * 0.75)
            self.sdr.pyhackrf_set_baseband_filter_bandwidth(bw)
            self.sdr.pyhackrf_start_rx()
        elif self.device_type == "rtlsdr":
            self.sdr.sample_rate = state['sample_rate']
        elif self.device_type == "bladerf":
            self.rx_ch.sample_rate = int(state['sample_rate'])
            self.rx_ch.bandwidth = int(state['sample_rate'] / 2)
            
        self.update_x_axis()

    def on_lna_changed(self, text):
        if not text: return
        val = int(text.replace(" dB", ""))
        if self.device_type == "hackrf":
            self.sdr.pyhackrf_set_lna_gain(val)
        elif self.device_type == "bladerf":
            self.rx_ch.gain = val

    def on_vga_changed(self, text):
        if self.device_type == "hackrf":
            val = int(text.replace(" dB", ""))
            self.sdr.pyhackrf_set_vga_gain(val)

    def on_fft_changed(self, text):
        state['fft_size'] = int(text)
        self.reset_traces()
        self.update_x_axis()

    def update_x_axis(self):
        cf = state['center_freq']
        sr = state['sample_rate']
        
        if state.get('demod_mode') == 'wbfm':
            fs = int(sr) # Eje X de 300,000 puntos (1 segundo)
        else:
            fs = state['fft_size'] # Eje X normal dictado por la interfaz
            
        self.f_axis = np.linspace(cf - sr/2, cf + sr/2, fs) / 1e6
        self.freq_plot.setXRange((cf - sr/2)/1e6, (cf + sr/2)/1e6)

    def closeEvent(self, event):
        print(f"Cerrando demodulador y apagando {self.device_type}...")
        try:
            if self.device_type == "hackrf":
                self.sdr.pyhackrf_stop_rx()
                self.sdr.pyhackrf_close()
                pyhackrf.pyhackrf_exit()
            elif self.device_type == "rtlsdr":
                self.sdr.cancel_read_async()
                self.sdr.close()
            elif self.device_type == "bladerf":
                self.bladerf_running = False # Frena el hilo worker (while)
                time.sleep(0.1) # Le da tiempo a terminar de leer el USB
                self.rx_ch.enable = False
                self.sdr.close()
        except: pass
        event.accept()

    def update_plot(self, PSD, raw_samples, PSD_audio=None, f_axis_audio=None):
        if self.is_paused:
            return
        
        if len(self.f_axis) == len(PSD):
            
            # --- LÓGICA DE TRACE ---
            display_psd = PSD # Por defecto, "White clear" usa la señal directa
            
            if self.trace_mode == "Max Hold":
                if self.max_hold_data is None or len(self.max_hold_data) != len(PSD):
                    self.max_hold_data = PSD.copy()
                else:
                    # Compara punto por punto y se queda con el más alto
                    self.max_hold_data = np.maximum(self.max_hold_data, PSD) 
                display_psd = self.max_hold_data

            elif self.trace_mode == "Average":
                if self.avg_buffer is None or self.avg_buffer.shape[1] != len(PSD):
                    # Matriz de 100 filas (muestras) x N columnas (frecuencias)
                    self.avg_buffer = np.zeros((self.AVG_MAX, len(PSD)))
                    self.avg_index = 0
                    self.avg_count = 0
                
                # Guarda la muestra actual en la posición del índice y avanza cíclicamente
                self.avg_buffer[self.avg_index] = PSD
                self.avg_index = (self.avg_index + 1) % self.AVG_MAX
                if self.avg_count < self.AVG_MAX:
                    self.avg_count += 1
                
                # Promedia sobre las muestras recolectadas hasta el momento
                display_psd = np.mean(self.avg_buffer[:self.avg_count], axis=0)

            # Graficamos la señal procesada
            self.freq_plot_curve.setData(self.f_axis, display_psd)
            
            # --- LÓGICA DE MARKERS Y DELTAS ---
            any_active = any(m['active'] for m in self.markers_info.values())
            if any_active:
                texto_global = ""
                current_frame_data = {} # Para guardar X e Y y poder restar

                # 1. Posicionar los gráficos de los que están activos
                for key, data in self.markers_info.items():
                    if data['active']:
                        idx = (np.abs(self.f_axis - data['freq'])).argmin()
                        x_val = self.f_axis[idx]
                        y_val = display_psd[idx]
                        data['item'].setData([x_val], [y_val])
                        current_frame_data[key] = {'x': x_val, 'y': y_val, 'color': data['color']}

                # 2. Armar el texto para M1 y Delta 1
                if 'M1' in current_frame_data:
                    m1 = current_frame_data['M1']
                    texto_global += f"<span style='color:{m1['color']}'><b>M1:</b> {m1['x']:.3f} MHz | {m1['y']:.2f} dB</span><br>"
                if 'D1' in current_frame_data:
                    d1 = current_frame_data['D1']
                    if 'M1' in current_frame_data: # Si M1 existe, Delta 1 es relativo a M1
                        dx = d1['x'] - m1['x']
                        dy = d1['y'] - m1['y']
                        texto_global += f"<span style='color:{d1['color']}'><b>Δ1:</b> {dx:+.3f} MHz | {dy:+.2f} dB</span><br>"
                    else: # Si prendieron D1 pero M1 está apagado, muestra valores absolutos
                        texto_global += f"<span style='color:{d1['color']}'><b>Δ1:</b> {d1['x']:.3f} MHz | {d1['y']:.2f} dB (Falta M1)</span><br>"

                # 3. Armar el texto para M2 y Delta 2
                if 'M2' in current_frame_data:
                    m2 = current_frame_data['M2']
                    texto_global += f"<span style='color:{m2['color']}'><b>M2:</b> {m2['x']:.3f} MHz | {m2['y']:.2f} dB</span><br>"
                if 'D2' in current_frame_data:
                    d2 = current_frame_data['D2']
                    if 'M2' in current_frame_data:
                        dx = d2['x'] - m2['x']
                        dy = d2['y'] - m2['y']
                        texto_global += f"<span style='color:{d2['color']}'><b>Δ2:</b> {dx:+.3f} MHz | {dy:+.2f} dB</span><br>"
                    else:
                        texto_global += f"<span style='color:{d2['color']}'><b>Δ2:</b> {d2['x']:.3f} MHz | {d2['y']:.2f} dB (Falta M2)</span><br>"

                # 4. Actualizar el cuadro
                self.marker_text_box.setHtml(texto_global)
                view_rect = self.freq_plot.viewRange()
                self.marker_text_box.setPos(view_rect[0][1], view_rect[1][1])
            
            if self.is_recording:
                self.recorded_samples.append(raw_samples.copy())

            # Graficar la FM demodulada si estamos en ese modo
            if state.get('demod_mode') == 'wbfm' and PSD_audio is not None:
                self.q2_curve.setData(f_axis_audio, PSD_audio)

class StartupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEMODULADOR")
        self.resize(QSize(400, 350))
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        layout = QVBoxLayout()
        
        label = QLabel("Dispositivos SDR detectados por USB:")
        label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(label)

        # Botón para forzar el escaneo manual
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

        self.main_app_window = None
        self.scan_devices() # Escaneo automático al abrir la ventana

    def scan_devices(self):
        self.device_list.clear()
        devices_found = 0

        # Buscar HackRF por su VID y PID
        try:
            if usb.core.find(idVendor=0x1d50, idProduct=0x6089):
                self.device_list.addItem("HackRF One")
                devices_found += 1
        except Exception: pass

        # Buscar RTL-SDR por su VID y PID
        try:
            if usb.core.find(idVendor=0x0bda, idProduct=0x2838):
                self.device_list.addItem("RTL-SDR")
                devices_found += 1
        except Exception: pass

        # Buscar Nuand bladeRF por su Vendor ID
        try:
            if usb.core.find(idVendor=0x2cf0) or usb.core.find(idVendor=0x1d50, idProduct=0x6066):
                self.device_list.addItem("Nuand bladeRF x40")
                devices_found += 1
        except Exception: pass

        if devices_found == 0:
            # 1. Creamos el ítem físicamente
            item = QListWidgetItem("⚠️ No se encontraron dispositivos SDR")
            # 2. Le sacamos la propiedad de ser seleccionable/clickeable
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            # 3. Lo metemos en la lista
            self.device_list.addItem(item)

    def launch_main_window(self, item):
        device_name = item.text()
        
        if "HackRF" in device_name:
            print("Iniciando HackRF...")
            pyhackrf.pyhackrf_init()
            sdr = pyhackrf.pyhackrf_open()
            sdr.pyhackrf_set_sample_rate(int(state['sample_rate']))
            bw_inicial = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(state['sample_rate'] * 0.75)
            sdr.pyhackrf_set_baseband_filter_bandwidth(bw_inicial)
            sdr.pyhackrf_set_freq(int(state['center_freq']))
            sdr.pyhackrf_set_lna_gain(32)
            sdr.pyhackrf_set_vga_gain(50)
            sdr.set_rx_callback(rx_callback)
            sdr.pyhackrf_start_rx()

            self.main_app_window = MainWindow(sdr, device_type="hackrf")
            self.main_app_window.show()
            self.close()

        elif "RTL-SDR" in device_name:
            print("Iniciando RTL-SDR...")
            if RtlSdr is None:
                print("Librería rtlsdr no está instalada.")
                return
            
            sdr = RtlSdr()
            # La RTL no soporta 10 MHz. Su máximo estable es 2.4 MHz.
            state['sample_rate'] = 2.4e6 
            sdr.sample_rate = state['sample_rate']
            sdr.center_freq = state['center_freq']
            sdr.gain = 'auto'

            #Callback especial para la RTL (ya entrega números complejos directos)
            def rtl_callback(samples, context):
                process_iq_samples(samples)

            self.main_app_window = MainWindow(sdr, device_type="rtlsdr")
            self.main_app_window.show()
            self.close()

            # La lectura de la RTL bloquea el programa, así que lo mandamos a un hilo secundario
            t = threading.Thread(target=sdr.read_samples_async, args=(rtl_callback, 8192))
            t.daemon = True
            t.start()
        
        elif "bladeRF" in device_name:
            print("Iniciando bladeRF...")
            if bladerf is None:
                print("Librería bladerf no está instalada.")
                return
            
            sdr = bladerf.BladeRF()
            rx_ch = sdr.Channel(bladerf.CHANNEL_RX(0))
            
            # Configuraciones iniciales (BladeRF soporta anchos de banda enormes)
            state['sample_rate'] = 20e6
            state['center_freq'] = 300e6
            rx_ch.frequency = int(state['center_freq'])
            rx_ch.sample_rate = int(state['sample_rate'])
            rx_ch.bandwidth = int(state['sample_rate'])
            rx_ch.gain = 0 # La bladeRF usa una sola ganancia global (-15 a 60 dB)

            # Setup del stream sincrónico de altísima velocidad
            sdr.sync_config(
                layout=bladerf._bladerf.ChannelLayout.RX_X1,
                fmt=bladerf._bladerf.Format.SC16_Q11,
                num_buffers=16,
                buffer_size=32768,
                num_transfers=8,
                stream_timeout=3500
            )
            rx_ch.enable = True

            self.main_app_window = MainWindow(sdr, device_type="bladerf")
            self.main_app_window.bladerf_running = True
            self.main_app_window.show()
            self.close()

            # El hilo secundario que succiona datos del USB a lo bestia
            def bladerf_worker(app_window, device):
                bytes_per_sample = 4 # SC16_Q11 = 2 enteros de 16 bits (I y Q)
                buf_size = 32768
                buf = bytearray(buf_size * bytes_per_sample)
                emit_counter = 0
                
                while app_window.bladerf_running:
                    try:
                        # Leemos los datos para vaciar el USB
                        device.sync_rx(buf, buf_size)
                        
                        # Conversión directa usando numpy (súper optimizado)
                        data = np.frombuffer(buf, dtype=np.int16)
                        # Dividimos por 2048 porque el formato usa rango [-2048 a 2047]
                        c_samples = (data[0::2] + 1j * data[1::2]) / 2048.0 
                        
                        if state.get('demod_mode') == 'wbfm':
                            # En demodulación mandamos TODO para no perder continuidad en el buffer de 1s
                            process_iq_samples(c_samples)
                        else:
                            # En modo normal saltamos la graficación para no ahogar a PyQt con FPS absurdos
                            emit_counter += 1
                            if emit_counter % 20 == 0:
                                process_iq_samples(c_samples)
                                
                    except Exception as e:
                        print("Error en rx bladeRF:", e)
                        break

            t = threading.Thread(target=bladerf_worker, args=(self.main_app_window, sdr))
            t.daemon = True
            t.start()

# --- INICIALIZACIÓN DE LA APP ---
app = QApplication([])

startup_window = StartupWindow()
startup_window.show()

signal.signal(signal.SIGINT, signal.SIG_DFL)
app.exec()