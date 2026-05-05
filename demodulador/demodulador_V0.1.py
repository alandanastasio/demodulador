### DEMODULADOR V0.2 ###

from PyQt6.QtCore import QSize, Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QFormLayout, 
                             QToolBar, QToolButton, QMenu, QFileDialog)
import pyqtgraph as pg
import numpy as np
import signal
from python_hackrf import pyhackrf
import time
import datetime

# Estado global para compartir entre la GUI y el hilo de C (callback)
state = {
    'fft_size': 4096,
    'center_freq': 100e6,
    'sample_rate': 10e6
}

class SignalEmitter(QObject):
    data_updated = pyqtSignal(np.ndarray, np.ndarray)

emitter = SignalEmitter()

def rx_callback(device, buffer, buffer_length, valid_length):
    accepted_samples = buffer[:valid_length].astype(np.int8)
    c_samples = (accepted_samples[0::2] + 1j * accepted_samples[1::2]) / 128.0

    fs = state['fft_size']
    if len(c_samples) >= fs:
        chunk = c_samples[:fs].copy() # Sacamos una copia solo para el gráfico
        chunk = chunk - np.mean(chunk)

        potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk)))**2 / fs
        PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))

        centro = fs // 2
        PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0

        # Emitimos el PSD para la pantalla y TODAS las muestras para grabar
        emitter.data_updated.emit(PSD, c_samples)

    return 0

class MainWindow(QMainWindow):
    def __init__(self, sdr_device):
        super().__init__()
        self.sdr = sdr_device
        self.setWindowTitle("Demodulador")
        self.resize(QSize(1200, 600))
        self.setMinimumSize(QSize(800, 400))

        # Variables para la grabación y reproducción
        self.is_recording = False
        self.recorded_samples = []
        self.playback_timer = QTimer()                            
        self.playback_timer.timeout.connect(self.playback_step)   
        self.playback_data = None                                 
        self.playback_index = 0
        self.is_looping = False                                   

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

        # 2. Crear el Menú que va a contener las opciones
        self.rec_play_menu = QMenu()

        # 3. Crear las acciones (Opciones del menú)
        self.record_action = QAction("🔴 Iniciar Grabación", self)
        self.record_action.triggered.connect(self.toggle_recording)
        
        self.play_action = QAction("▶ Reproducir Archivo", self)
        self.play_action.triggered.connect(lambda: self.load_and_play(loop=False))
        
        # --- NUEVOS BOTONES ---
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


        # Layout Principal
        main_layout = QHBoxLayout()
        
        # --- LADO IZQUIERDO: GRÁFICO ---
        self.freq_plot = pg.PlotWidget(labels={'left': 'Potencia [dB]', 'bottom': 'Frecuencia [MHz]'})
        self.freq_plot.setMouseEnabled(x=True, y=True)
        self.freq_plot_curve = self.freq_plot.plot([], pen=pg.mkPen(color='#FFD500', width=1.5))
        self.freq_plot.setYRange(-70, 10)
        self.update_x_axis()
        
        main_layout.addWidget(self.freq_plot, stretch=4)

        # --- LADO DERECHO: CONTROLES ---
        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        form_layout = QFormLayout()

        self.freq_input = QDoubleSpinBox()
        self.freq_input.setSuffix(" MHz")
        self.freq_input.setDecimals(3)
        self.freq_input.setRange(1.0, 6000.0)
        self.freq_input.setSingleStep(0.1)
        self.freq_input.setValue(state['center_freq'] / 1e6)
        self.freq_input.valueChanged.connect(self.on_freq_changed)
        form_layout.addRow(QLabel("FREQ CENTRAL:"), self.freq_input)

        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["2 MHz", "4 MHz", "8 MHz", "10 MHz", "12.5 MHz", "16 MHz", "20 MHz"])
        self.sr_combo.setCurrentText("10 MHz")
        self.sr_combo.currentTextChanged.connect(self.on_sr_changed)
        form_layout.addRow(QLabel("SAMP RATE:"), self.sr_combo)

        self.lna_combo = QComboBox()
        self.lna_combo.addItems([f"{g} dB" for g in range(0, 48, 8)])
        self.lna_combo.setCurrentText("32 dB")
        self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
        form_layout.addRow(QLabel("LNA GAIN:"), self.lna_combo)

        self.vga_combo = QComboBox()
        self.vga_combo.addItems([f"{g} dB" for g in range(0, 64, 2)])
        self.vga_combo.setCurrentText("50 dB")
        self.vga_combo.currentTextChanged.connect(self.on_vga_changed)
        form_layout.addRow(QLabel("VGA GAIN:"), self.vga_combo)

        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["512", "1024", "2048", "4096", "8192"])
        self.fft_combo.setCurrentText("4096")
        self.fft_combo.currentTextChanged.connect(self.on_fft_changed)
        form_layout.addRow(QLabel("TAMAÑO FFT:"), self.fft_combo)

        controls_layout.addLayout(form_layout)
        
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedWidth(300)
        main_layout.addWidget(controls_widget, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        emitter.data_updated.connect(self.update_plot)

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
        self.freq_plot_curve.setPen(pg.mkPen(color="#22FF00", width=1.5))

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

    def on_freq_changed(self, val_mhz):
        state['center_freq'] = val_mhz * 1e6
        self.sdr.pyhackrf_set_freq(int(state['center_freq']))
        self.update_x_axis()

    def on_sr_changed(self, text):
        val_mhz = float(text.replace(" MHz", ""))
        state['sample_rate'] = val_mhz * 1e6
        
        self.sdr.pyhackrf_stop_rx()

        self.sdr.pyhackrf_set_sample_rate(int(state['sample_rate']))
        bw = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(state['sample_rate'] * 0.75)
        self.sdr.pyhackrf_set_baseband_filter_bandwidth(bw)
        
        self.update_x_axis()
        self.sdr.pyhackrf_start_rx()

    def on_lna_changed(self, text):
        val = int(text.replace(" dB", ""))
        self.sdr.pyhackrf_set_lna_gain(val)

    def on_vga_changed(self, text):
        val = int(text.replace(" dB", ""))
        self.sdr.pyhackrf_set_vga_gain(val)

    def on_fft_changed(self, text):
        state['fft_size'] = int(text)
        self.update_x_axis()

    def update_x_axis(self):
        cf = state['center_freq']
        sr = state['sample_rate']
        fs = state['fft_size']
        self.f_axis = np.linspace(cf - sr/2, cf + sr/2, fs) / 1e6
        self.freq_plot.setXRange((cf - sr/2)/1e6, (cf + sr/2)/1e6)

    def update_plot(self, PSD, raw_samples):
        if len(self.f_axis) == len(PSD):
            self.freq_plot_curve.setData(self.f_axis, PSD)
            if self.is_recording:
                # Guardamos las muestras complejas (IQ) intactas
                self.recorded_samples.append(raw_samples.copy())

# --- INICIALIZACIÓN HACKRF Y APP ---
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

app = QApplication([])
window = MainWindow(sdr)
window.show()

signal.signal(signal.SIGINT, signal.SIG_DFL)
app.exec()

sdr.pyhackrf_stop_rx()
sdr.pyhackrf_close()
pyhackrf.pyhackrf_exit()