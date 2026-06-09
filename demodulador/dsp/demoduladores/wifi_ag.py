import numpy as np
import threading
from scipy.ndimage import uniform_filter1d
from .base import DemoduladorBase

# --- SCHMIDL & COX ---
def schmidl_cox_metric(iq_signal, N=16, W=64):
    # 1. Quitamos el DC Offset SOLO localmente para destruir la falsa correlación del hardware
    iq_clean = iq_signal - np.mean(iq_signal)
    
    # 2. Productos cruzados y energía
    prod = np.conj(iq_clean[:-N]) * iq_clean[N:]
    energy = np.abs(iq_clean[N:]) ** 2
    
    # 3. Integración sobre una ventana LARGA (W=64). 
    # Esto aplasta el ruido a 0 y mantiene el STS en 1.
    ventana = np.ones(W)
    P = np.convolve(prod, ventana, mode='valid')
    R = np.convolve(energy, ventana, mode='valid')
    
    # 4. El silenciador de ruido absoluto (usando el pico máximo)
    max_energia = np.max(R)
    mascara_energia = R > (0.2 * max_energia)
    
    # 5. Métrica final
    M = np.abs(P) ** 2 / (R ** 2 + 1e-10)
    M[~mascara_energia] = 0.0 
    
    return M, P, R

class DemoduladorWiFiAG(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 20e6 
        self.fft_size = 4096
        self.buffer_medicion = []
        self.muestras_acumuladas = 0
        self.is_processing = False
        self.last_heavy_results = {}
        self.nuevos_datos_listos = False

    @property
    def id(self): return "wifi_ag"

    @property
    def nombre_mostrar(self): return "WiFi 802.11a/g (OFDM)"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.buffer_medicion = []
        self.muestras_acumuladas = 0
        self.is_processing = False

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        self.buffer_medicion.append(muestras_iq)
        self.muestras_acumuladas += len(muestras_iq)
        
        muestras_necesarias = int(self.sample_rate * 0.1)
        if self.muestras_acumuladas >= muestras_necesarias:
            if not self.is_processing:
                bloque_iq = np.concatenate(self.buffer_medicion)[:muestras_necesarias]
                self.is_processing = True
                threading.Thread(target=self._procesar_fondo, args=(bloque_iq,)).start()
                
            self.buffer_medicion = []
            self.muestras_acumuladas = 0

        if self.nuevos_datos_listos:
            self.nuevos_datos_listos = False
            return self.last_heavy_results
            
        return None

    def _procesar_fondo(self, bloque_iq: np.ndarray):
        try:
            fs = self.fft_size
            
            # 1. BÚSQUEDA GRUESA (Energía)
            energia = np.abs(bloque_iq) ** 2
            energia_suave = uniform_filter1d(energia, size=50)
            max_energia = np.max(energia_suave)
            
            chunk_trigger = None
            metrica_sc = None # Aquí guardaremos la curva para graficarla
            
            if max_energia > 0:
                energia_norm = energia_suave / max_energia
                en_burst = energia_norm > 0.3
                cambios = np.diff(en_burst.astype(int))
                inicios_burst = np.where(cambios == 1)[0]
                
                if len(inicios_burst) > 0:
                    for inicio in inicios_burst:
                        # Extraemos un "barrio" alrededor de donde saltó la energía
                        # 200 muestras antes y 2000 después para asegurar que captamos el preámbulo
                        ini_ext = max(0, inicio - 200)
                        fin_ext = min(len(bloque_iq), inicio + 2000)
                        segmento = bloque_iq[ini_ext:fin_ext]
                        
                        if len(segmento) < 64:
                            continue
                          
                        # 2. BÚSQUEDA FINA (Schmidl & Cox con el silenciador de energía)
                        M, P, R = schmidl_cox_metric(segmento, N=16)
                        if len(M) == 0: continue
                        
                        # Buscamos el índice exacto donde la correlación absoluta supera 0.7
                        indices_sts = np.where(M > 0.7)[0]
                        
                        if len(indices_sts) > 0:
                            muestra_local = indices_sts[0]
                            
                            # Compensamos la anticipación de la ventana de 64 muestras
                            offset_calibracion = 16 
                            sts_abs_idx = ini_ext + muestra_local + offset_calibracion
                            
                            margen_visual = 150
                            inicio_visual = max(0, sts_abs_idx - margen_visual)
                            
                            if inicio_visual + fs <= len(bloque_iq):
                                chunk_trigger = bloque_iq[inicio_visual : inicio_visual + fs].copy()
                                
                                # Extraemos el preámbulo para el Q3 (Las 400 muestras)
                                if sts_abs_idx + 400 <= len(bloque_iq):
                                    preamble_chunk = bloque_iq[sts_abs_idx : sts_abs_idx + 400]
                                    metrica_sc = np.abs(preamble_chunk) 
                                else:
                                    metrica_sc = None
                                
                                break

            # 3. CÁLCULO DE ESPECTRO
            chunk_psd = chunk_trigger if chunk_trigger is not None else bloque_iq[:fs].copy()
            chunk_psd = chunk_psd - np.mean(chunk_psd)
            potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk_psd)))**2 / fs
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            
            centro = fs // 2
            PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0
            
            self.last_heavy_results = {
                'psd_rf': PSD,
                'rf_chunk': chunk_trigger,
                # Envíamos la métrica usando esta variable puente para no romper main.py
                'mpx_time': metrica_sc 
            }
            self.nuevos_datos_listos = True
            
        finally:
            self.is_processing = False