
### DEMODULADOR V0.1 ###

from PyQt6.QtCore import QSize, Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QFormLayout, QPushButton)
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
    fft_updated = pyqtSignal(np.ndarray)

emitter = SignalEmitter()

def rx_callback(device, buffer, buffer_length, valid_length):
    # Extraer las muestras y convertirlas a formato complejo
    accepted_samples = buffer[:valid_length].astype(np.int8)
    c_samples = (accepted_samples[0::2] + 1j * accepted_samples[1::2]) / 128.0

    fs = state['fft_size']
    if len(c_samples) >= fs:
        chunk = c_samples[:fs]

        # Eliminar pico de DC (oscilador interno)
        chunk = chunk - np.mean(chunk)

        # Calcular PSD sin averageo y evitar log10(0)
        potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk)))**2 / fs
        PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))

        # --- NUEVO: Interpolar el pozo del centro (DC) ---
        centro = fs // 2
        PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0

        #time.sleep(0.1)

        # Emitir a la interfaz gráfica
        emitter.fft_updated.emit(PSD)

    return 0

class MainWindow(QMainWindow):
    def __init__(self, sdr_device):
        super().__init__()
        self.sdr = sdr_device
        self.setWindowTitle("Demodulador")
        self.setFixedSize(QSize(1200, 600))

        # Variables para la grabación
        self.is_recording = False
        self.recorded_psds = []

        # Layout Principal (Horizontal: Gráfico a la izquierda, Controles a la derecha)
        main_layout = QHBoxLayout()
        
        # --- LADO IZQUIERDO: GRÁFICO ---
        self.freq_plot = pg.PlotWidget(labels={'left': 'Potencia [dB]', 'bottom': 'Frecuencia [MHz]'})
        self.freq_plot.setMouseEnabled(x=False, y=True)
        self.freq_plot_curve = self.freq_plot.plot([])
        self.freq_plot.setYRange(-70, 10)
        self.update_x_axis()
        
        main_layout.addWidget(self.freq_plot, stretch=4) # Ocupa más espacio

        # --- LADO DERECHO: CONTROLES ---
        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        form_layout = QFormLayout()

        # Botón de Grabación
        self.record_btn = QPushButton("🔴 INICIAR GRABACIÓN")
        self.record_btn.setStyleSheet("background-color: #8b0000; color: white; font-weight: bold; padding: 10px;")
        self.record_btn.clicked.connect(self.toggle_recording)
        form_layout.addRow(self.record_btn)

        # 1. Frecuencia Central (Casilla con saltos de 0.1 MHz)
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setSuffix(" MHz")
        self.freq_input.setDecimals(3)
        self.freq_input.setRange(1.0, 6000.0) # Rango de la HackRF
        self.freq_input.setSingleStep(0.1)
        self.freq_input.setValue(state['center_freq'] / 1e6)
        self.freq_input.valueChanged.connect(self.on_freq_changed)
        form_layout.addRow(QLabel("FREQ CENTRAL:"), self.freq_input)

        # 2. Sample Rate (Desplegable)
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["2 MHz", "4 MHz", "8 MHz", "10 MHz", "12.5 MHz", "16 MHz", "20 MHz"])
        self.sr_combo.setCurrentText("10 MHz")
        self.sr_combo.currentTextChanged.connect(self.on_sr_changed)
        form_layout.addRow(QLabel("SAMP RATE:"), self.sr_combo)

        # 3. LNA Gain (Desplegable 0 a 40 dB en saltos de 8 dB)
        self.lna_combo = QComboBox()
        self.lna_combo.addItems([f"{g} dB" for g in range(0, 48, 8)])
        self.lna_combo.setCurrentText("32 dB")
        self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
        form_layout.addRow(QLabel("LNA GAIN:"), self.lna_combo)

        # 4. VGA Gain (Desplegable 0 a 62 dB en saltos de 2 dB)
        self.vga_combo = QComboBox()
        self.vga_combo.addItems([f"{g} dB" for g in range(0, 64, 2)])
        self.vga_combo.setCurrentText("50 dB")
        self.vga_combo.currentTextChanged.connect(self.on_vga_changed)
        form_layout.addRow(QLabel("VGA GAIN:"), self.vga_combo)

        # 5. Tamaño FFT (Desplegable)
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["512", "1024", "2048", "4096", "8192"])
        self.fft_combo.setCurrentText("4096")
        self.fft_combo.currentTextChanged.connect(self.on_fft_changed)
        form_layout.addRow(QLabel("TAMAÑO FFT:"), self.fft_combo)

        controls_layout.addLayout(form_layout)
        
        # Contenedor para el layout de controles
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedWidth(250) # Ancho fijo para el panel derecho
        main_layout.addWidget(controls_widget, stretch=1)

        # Configurar widget central
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Conectar actualización del gráfico
        emitter.fft_updated.connect(self.update_plot)

    def toggle_recording(self):
        if not self.is_recording:
            # Iniciar grabación
            self.is_recording = True
            self.recorded_psds = []
            self.record_btn.setText("⏹ DETENER Y GUARDAR")
            self.record_btn.setStyleSheet("background-color: #006400; color: white; font-weight: bold; padding: 10px;")
            print("Grabación iniciada...")
            
            # Deshabilitar controles que alteren la metadata de la grabación en curso
            self.freq_input.setEnabled(False)
            self.sr_combo.setEnabled(False)
            self.fft_combo.setEnabled(False)
        else:
            # Detener y guardar
            self.is_recording = False
            self.record_btn.setText("🔴 INICIAR GRABACIÓN")
            self.record_btn.setStyleSheet("background-color: #8b0000; color: white; font-weight: bold; padding: 10px;")
            
            # Rehabilitar controles
            self.freq_input.setEnabled(True)
            self.sr_combo.setEnabled(True)
            self.fft_combo.setEnabled(True)

            if len(self.recorded_psds) > 0:
                # Generar nombre de archivo con timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"espectro_{timestamp}.npz"
                
                # Guardar el array 2D (tiempo x frecuencias) comprimido junto a su metadata
                np.savez_compressed(
                    filename,
                    psd_data=np.array(self.recorded_psds),
                    center_freq=state['center_freq'],
                    sample_rate=state['sample_rate'],
                    fft_size=state['fft_size']
                )
                print(f"Grabación guardada exitosamente en: {filename}")
                print(f"Paquetes grabados: {len(self.recorded_psds)}")
                self.recorded_psds = [] # Limpiar memoria

    # --- SLOTS DE ACTUALIZACIÓN (CONTROLES A HACKRF) ---
    def on_freq_changed(self, val_mhz):
        state['center_freq'] = val_mhz * 1e6
        self.sdr.pyhackrf_set_freq(int(state['center_freq']))
        self.update_x_axis()

    def on_sr_changed(self, text):
        val_mhz = float(text.replace(" MHz", ""))
        state['sample_rate'] = val_mhz * 1e6
        
        # 1. Frenar la recepción de datos por USB
        self.sdr.pyhackrf_stop_rx()

        # 2. Aplicar los cambios de hardware críticos
        self.sdr.pyhackrf_set_sample_rate(int(state['sample_rate']))
        bw = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(state['sample_rate'] * 0.75)
        self.sdr.pyhackrf_set_baseband_filter_bandwidth(bw)
        
        # 3. Recalcular el eje X para el gráfico
        self.update_x_axis()

        # 4. Volver a arrancar la recepción
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
        # Recalcular eje X cuando cambian frecuencia, SR o tamaño FFT
        cf = state['center_freq']
        sr = state['sample_rate']
        fs = state['fft_size']
        self.f_axis = np.linspace(cf - sr/2, cf + sr/2, fs) / 1e6
        self.freq_plot.setXRange((cf - sr/2)/1e6, (cf + sr/2)/1e6)

    def update_plot(self, PSD):
        if len(self.f_axis) == len(PSD):
            self.freq_plot_curve.setData(self.f_axis, PSD)
            # Acumular los datos si estamos grabando
            if self.is_recording:
                self.recorded_psds.append(PSD.copy())


# --- INICIALIZACIÓN HACKRF Y APP ---
pyhackrf.pyhackrf_init()
sdr = pyhackrf.pyhackrf_open()

# Valores iniciales en hardware
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