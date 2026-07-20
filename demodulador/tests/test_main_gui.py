import numpy as np
from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg
import sys
import os

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

import main

class DummyRadio:
    nombre = "DUMMY"
    def set_sample_rate(self, sr): pass
    def set_muestras_por_bloque(self, m): pass
    def set_freq(self, m): pass
    def configurar(self, a, b): pass
    def start_rx(self, *args): pass

window = main.MainWindow(DummyRadio())
window.set_wifi_ag_mode()

resultados = {
    'mpx_time': np.array([1, 2, 3]),
    'audio_time_L': np.array([0.1, 0.2]),
    'audio_time_R': np.array([0.1, 0.2]),
    'psd_rf': np.array([0.1, 0.2]),
    'rf_chunk': np.array([0.1, 0.2]),
    'metricas': {'wifi_metrics': {'mod': '16-QAM', 'mbps': 24}},
    'evm_data': {
        'subc_x': np.arange(48),
        'subc_rms': np.zeros(48),
        'subc_peak': np.zeros(48),
        'sym_rms': np.zeros(10),
        'sym_peak': np.zeros(10)
    }
}

try:
    window.data_updated(resultados)
    print("data_updated SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()

