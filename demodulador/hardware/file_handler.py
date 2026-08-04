import numpy as np
import threading
import time
from .sdr_base import SDRBase

class FileHandler(SDRBase):
    def __init__(self, rx_callback, file_path: str):
        super().__init__(rx_callback)
        self.file_path = file_path
        
        self.data = np.fromfile(file_path, dtype=np.complex64)
        print(f"Cargado {file_path}: {len(self.data)} muestras")
        
        self._thread = None
        self.muestras_por_bloque = 4096 
        self.sample_rate = 3.84e6 

    @property
    def nombre(self):
        return f"File ({self.file_path.split('/')[-1]})"

    def configurar(self, sample_rate: float, center_freq: float):
        self.set_sample_rate(sample_rate)
        self.set_freq(center_freq)

    def set_freq(self, freq_hz: float):
        pass

    def set_sample_rate(self, sr_hz: float):
        self.sample_rate = sr_hz
        self.muestras_por_bloque = int(sr_hz * 0.002)

    def set_gain(self, gain_db: int):
        pass
        
    def set_muestras_por_bloque(self, m: int):
        self.muestras_por_bloque = m

    def start_rx(self):
        self.is_running = True
        self._thread = threading.Thread(target=self._rx_worker, daemon=True)
        self._thread.start()

    def stop_rx(self):
        self.is_running = False
        if self._thread:
            self._thread.join()

    def _rx_worker(self):
        idx = 0
        N = len(self.data)
        
        while self.is_running:
            chunk_size = self.muestras_por_bloque
            if idx + chunk_size > N:
                # Wrap around
                chunk = np.concatenate((self.data[idx:N], self.data[0:chunk_size - (N - idx)]))
                idx = chunk_size - (N - idx)
            else:
                chunk = self.data[idx : idx + chunk_size]
                idx += chunk_size
                
            if self.rx_callback:
                self.rx_callback(chunk.copy())
            
            # Simulamos tiempo real
            time.sleep(chunk_size / self.sample_rate)
            
    def close(self):
        self.stop_rx()
