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
        
    def procesar(self, muestras_iq: np.ndarray) -> dict:
        with self._lock:
            if self.nuevos_datos_listos:
                self.nuevos_datos_listos = False
                return self.last_heavy_results

        if muestras_iq is None or len(muestras_iq) == 0:
            return None
            
        with self._lock:
            self.buffer_medicion.append(muestras_iq)
            self.muestras_acumuladas += len(muestras_iq)
            
        ahora = time.time()
        if not self.is_processing and ahora >= self.proxima_captura:
            min_muestras = int(0.02 * self.sample_rate)
            if self.muestras_acumuladas >= min_muestras:
                with self._lock:
                    chunk = np.concatenate(self.buffer_medicion)
                    chunk_procesar = chunk[:min_muestras]
                    sobrante = chunk[min_muestras:]
                    self.buffer_medicion = [sobrante] if len(sobrante) > 0 else []
                    self.muestras_acumuladas = len(sobrante)
                    
                    self.is_processing = True
                    threading.Thread(target=self._procesar_heavy_thread, args=(chunk_procesar,), daemon=True).start()
                    
        return None

    def _procesar_heavy_thread(self, chunk):
        try:
            self.ultimo_chunk_norm = chunk / np.max(np.abs(chunk))
            
            # TODO: Implementar sincronización Schmidl & Cox, SC-FDMA, etc.
            
            # --- Cálculo básico de espectro para UI ---
            ui_fs = self.fft_size
            rf_chunk_ui = chunk[:ui_fs] if len(chunk) >= ui_fs else np.pad(chunk, (0, ui_fs - len(chunk)))
            
            chunk_psd = chunk.copy()[:ui_fs]
            if len(chunk_psd) < ui_fs:
                chunk_psd = np.pad(chunk_psd, (0, ui_fs - len(chunk_psd)))
                
            chunk_psd = chunk_psd - np.mean(chunk_psd)
            potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk_psd, n=ui_fs)))**2 / ui_fs
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            
            centro_psd = ui_fs // 2
            PSD[centro_psd] = (PSD[centro_psd - 1] + PSD[centro_psd + 1]) / 2.0
            
            resultados = {
                'psd_rf': PSD,
                'rf_chunk': rf_chunk_ui, 
                'mpx_time': np.array([]),  
                'audio_time_L': np.array([]),
                'audio_time_R': np.array([]),
                'psd_mpx': np.array([]),
                'f_axis_mpx': np.array([]),
                'metricas': {
                    'lte_metrics': self.ultimo_lte_metrics,
                },
                'evm_data': None
            }

            with self._lock:
                self.last_heavy_results = resultados
                self.nuevos_datos_listos = True
                
        except Exception as e:
            print(f"Error en _procesar_heavy_thread Uplink: {e}")
        finally:
            self.is_processing = False
            self.proxima_captura = time.time() + self.pausa_entre_snapshots
