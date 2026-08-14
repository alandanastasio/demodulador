import numpy as np
import threading
import time
from .base import DemoduladorBase
from scipy import signal
from .lte_downlink import generar_secuencia_gold # Importamos funciones útiles del downlink si es necesario

class DemoduladorLTEUplink(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 30.72e6 
        self.fft_size = 2048
        
        self.Tu = 2048
        self.cp_len_1 = 160
        self.cp_len_2 = 144
        
        self.buffer_medicion = []
        self.muestras_acumuladas = 0
        self.is_processing = False
        
        self.last_heavy_results = {}
        self.nuevos_datos_listos = False
        self._lock = threading.Lock()
        self.pausa_entre_snapshots = 0.05
        self.proxima_captura = 0.0
        
        self.ultimo_chunk_norm = None
        self.ultimo_lte_metrics = {}
        self.ultimo_puntos_corr = np.array([])
        
        self.occupied_subcarriers = np.array([])
        self.rb_count = 100

    @property
    def id(self): return "lte_uplink"

    @property
    def nombre_mostrar(self): return "LTE Uplink (SC-FDMA)"

    def configurar(self, sample_rate: float, fft_size: int, rb_count: int = 100):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.rb_count = rb_count
            
        self.Tu = self.fft_size
        self.cp_len_1 = int(np.round(5.2e-6 * sample_rate))
        self.cp_len_2 = int(np.round(4.69e-6 * sample_rate))
        
        num_subcarriers = rb_count * 12
        start_idx = (self.Tu - num_subcarriers) // 2
        self.occupied_subcarriers = np.arange(start_idx, start_idx + num_subcarriers)
        
    def procesar_bloque(self, iq_data: np.ndarray) -> np.ndarray:
        if iq_data is None or len(iq_data) == 0:
            return iq_data
            
        with self._lock:
            self.buffer_medicion.append(iq_data)
            self.muestras_acumuladas += len(iq_data)
            
        if not self.is_processing and time.time() >= self.proxima_captura:
            with self._lock:
                min_muestras = int(0.02 * self.sample_rate)
                if self.muestras_acumuladas >= min_muestras:
                    chunk = np.concatenate(self.buffer_medicion)
                    self.buffer_medicion = []
                    self.muestras_acumuladas = 0
                    
                    self.is_processing = True
                    threading.Thread(target=self._procesar_heavy_thread, args=(chunk,), daemon=True).start()
                    
        return iq_data

    def _procesar_heavy_thread(self, chunk):
        try:
            self.ultimo_chunk_norm = chunk / np.max(np.abs(chunk))
            
            # Sincronización Schmidl & Cox para Uplink
            Tu = self.Tu
            cp_len = self.cp_len_2
            
            # TODO: Completar el resto de la implementación de Schmidl & Cox y DFT-S-OFDM
            
            # DMRS (3er símbolo del slot)
            # mia0 = self.cp_len_1 + Tu + 2*(self.cp_len_2 + Tu) + self.cp_len_2
            
            # PUSCH (otros símbolos)
            
            with self._lock:
                self.last_heavy_results = {
                    'chunk_norm': self.ultimo_chunk_norm,
                    'lte_metrics': self.ultimo_lte_metrics,
                }
                self.nuevos_datos_listos = True
                
        except Exception as e:
            print(f"Error en _procesar_heavy_thread Uplink: {e}")
        finally:
            self.is_processing = False
            self.proxima_captura = time.time() + self.pausa_entre_snapshots

    def get_resultados(self):
        with self._lock:
            if self.nuevos_datos_listos:
                res = self.last_heavy_results
                self.nuevos_datos_listos = False
                return res
            return None
