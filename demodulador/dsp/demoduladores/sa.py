import numpy as np
from .base import DemoduladorBase

import time

class SpectrumAnalyzer(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 10e6
        self.fft_size = 4096
        self.last_update = 0

    @property
    def id(self): return "sa"

    @property
    def nombre_mostrar(self): return "Analizador de Espectro (Sin Demodular)"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        """
        Recibe muestras crudas y devuelve únicamente el espectro RF.
        No genera audio ni métricas.
        """
        now = time.time()
        if getattr(self, 'last_update', 0) and now - self.last_update < 1.0 / 30.0:
            return None
        self.last_update = now
        
        fs = self.fft_size
        
        # Necesitamos tener al menos suficientes muestras para una FFT
        if len(muestras_iq) < fs:
            return None
            
        # Agarramos el chunk necesario
        chunk = muestras_iq[:fs].copy()
        
        # Removemos la componente de continua (DC Offset)
        chunk = chunk - np.mean(chunk)
        
        # Calculamos la FFT y la Potencia
        potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk)))**2 / fs
        
        # Convertimos a decibelios (dB) y evitamos el logaritmo de cero
        PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
        
        # Suavizamos el pico de continua (el centro exacto de la FFT en los SDRs
        # suele tener un pico irreal debido al hardware, así que lo promediamos)
        centro = fs // 2
        PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0
        
        # Empaquetamos y devolvemos solo lo que importa
        return {
            'psd_rf': PSD,
            'rf_chunk': chunk,
            # Como no hay audio ni métricas, devolvemos None en el resto
            'psd_mpx': None,
            'f_axis_mpx': None,
            'audio_time': None,
            't_axis_audio': None,
            'audio_out': None,
            'metricas': None
        }