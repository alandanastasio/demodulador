import numpy as np
from .base import DemoduladorBase

import time

class SpectrumAnalyzer(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 10e6
        self.fft_size = 4096
        self.last_update = 0
        self.window_type = 'rectangular'
        self._window_cache = None
        self._window_size = 0

    @property
    def id(self): return "sa"

    @property
    def nombre_mostrar(self): return "Analizador de Espectro (Sin Demodular)"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        
    def set_window(self, window_type: str):
        self.window_type = window_type
        self._window_cache = None
        
    def _get_window(self, N):
        if self._window_size == N and self._window_cache is not None:
            return self._window_cache
            
        self._window_size = N
        wt = self.window_type
        
        if wt == 'rectangular':
            self._window_cache = np.ones(N)
        elif wt == 'hanning':
            self._window_cache = np.hanning(N)
        elif wt == 'hamming':
            self._window_cache = np.hamming(N)
        elif wt == 'blackman':
            self._window_cache = np.blackman(N)
        elif wt == 'bartlett':
            self._window_cache = np.bartlett(N)
        elif wt == 'flattop':
            a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
            n = np.arange(N)
            # Evitar división por cero si N=1
            den = max(1, N - 1)
            self._window_cache = (a[0] 
                                - a[1]*np.cos(2*np.pi*n/den) 
                                + a[2]*np.cos(4*np.pi*n/den) 
                                - a[3]*np.cos(6*np.pi*n/den) 
                                + a[4]*np.cos(8*np.pi*n/den))
        else:
            self._window_cache = np.ones(N)
            
        return self._window_cache

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        """
        Recibe muestras crudas y devuelve únicamente el espectro RF.
        No genera audio ni métricas.
        """
        if muestras_iq is None:
            return None
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
        
        # Aplicamos la ventana
        ventana = self._get_window(fs)
        chunk_ventaneado = chunk * ventana
        
        # Calculamos la FFT y la Potencia
        potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk_ventaneado)))**2 / fs
        
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