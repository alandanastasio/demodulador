import numpy as np
import threading
from scipy.ndimage import uniform_filter1d
from .base import DemoduladorBase

# Constantes del preámbulo 802.11a/g (en muestras a 20 MHz)
_SC_N = 16   # Desplazamiento de la correlación S&C (mitad del STS period = 16 muestras)
_SC_W = 64   # Ventana de integración S&C

# --- SCHMIDL & COX ---
def schmidl_cox_metric(iq_signal, N=_SC_N, W=_SC_W):
    # 1. Quitamos el DC Offset SOLO localmente para destruir la falsa correlación del hardware
    iq_clean = iq_signal - np.mean(iq_signal)
    L = len(iq_signal)
    
    # 2. Productos cruzados y energía
    prod = np.conj(iq_clean[:-N]) * iq_clean[N:]
    energy = np.abs(iq_clean[N:]) ** 2
    
    # 3. Integración sobre una ventana LARGA (W=64). 
    # Esto aplasta el ruido a 0 y mantiene el STS en 1.
    ventana = np.ones(W)
    P = np.convolve(prod, ventana, mode='valid')
    R = np.convolve(energy, ventana, mode='valid')
    P = P[:L - 2 * N]
    R = R[:L - 2 * N]
    # 4. El silenciador de ruido absoluto (usando el pico máximo)
    max_energia = np.max(R)
    mascara_energia = R > (0.5 * max_energia)
    
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
        self._lock = threading.Lock()  # Protege last_heavy_results y nuevos_datos_listos

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
        # Descartamos cualquier resultado anterior para no mostrar datos de otra
        # configuración/sesión al arrancar.
        with self._lock:
            self.nuevos_datos_listos = False
            self.last_heavy_results = {}

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        self.buffer_medicion.append(muestras_iq)
        self.muestras_acumuladas += len(muestras_iq)
        
        muestras_necesarias = int(self.sample_rate * 0.1)
        if self.muestras_acumuladas >= muestras_necesarias:
            if not self.is_processing:
                bloque_iq = np.concatenate(self.buffer_medicion)[:muestras_necesarias]
                # Reseteamos el buffer SOLO cuando aceptamos el bloque para procesar.
                # Si is_processing está activo, seguimos acumulando para no perder bursts.
                self.buffer_medicion = []
                self.muestras_acumuladas = 0
                self.is_processing = True
                threading.Thread(target=self._procesar_fondo, args=(bloque_iq,), daemon=True).start()

        with self._lock:
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
            envolvente_preambulo = None  # |preámbulo| para visualizar estructura STS/LTS en Q3
            
            if max_energia > 0:
                energia_norm = energia_suave / max_energia
                en_burst = energia_norm > 0.3
                cambios = np.diff(en_burst.astype(int))
                inicios_burst = np.where(cambios == 1)[0]
                
                if len(inicios_burst) > 0:
                    for inicio in inicios_burst:
                        # Extraemos un "barrio" alrededor de donde saltó la energía.
                        # 200 muestras antes y 2000 después para asegurar que captamos el preámbulo.
                        ini_ext = max(0, inicio)
                        fin_ext = min(len(bloque_iq), inicio + 2000)
                        segmento = bloque_iq[ini_ext:fin_ext]
                        
                        if len(segmento) < _SC_W:
                            continue
                          
                        # 2. BÚSQUEDA FINA (Schmidl & Cox con silenciador de energía)
                        M, P, R = schmidl_cox_metric(segmento)
                        if len(M) == 0:
                            continue
                        
                        # Buscamos el primer índice donde la correlación normalizada supera 0.7
                        indices_sts = np.where(M > 0.7)[0]
                        
                        if len(indices_sts) > 0:
                            muestra_local = indices_sts[0]
                            
                            # La ventana de integración de W muestras introduce un retardo.
                            # Compensamos con _SC_N para apuntar al inicio real del STS.
                            sts_abs_idx = ini_ext + muestra_local + 10
                            
                            margen_visual = 150
                            inicio_visual = max(0, sts_abs_idx - margen_visual)
                            
                            if inicio_visual + fs <= len(bloque_iq):
                                chunk_trigger = bloque_iq[inicio_visual : inicio_visual + fs].copy()

                                # Eliminamos DC antes de la corrección de fase
                                chunk_trigger = chunk_trigger - np.mean(chunk_trigger)
                                
                                # --- ESTIMACIÓN Y CORRECCIÓN DE CFO GRUESO ---
                                # El ángulo de P acumula la rotación de fase que ocurrió
                                # en _SC_N muestras a causa del error de frecuencia (CFO).
                                fase_N_muestras = np.angle(P[muestra_local])
                                desfase_por_muestra = fase_N_muestras / _SC_N
                                
                                # n=0 apunta al primer instante del STS dentro del chunk_trigger
                                n_array = np.arange(len(chunk_trigger)) - margen_visual
                                chunk_trigger = chunk_trigger * np.exp(-1j * desfase_por_muestra * n_array)

                                # Extraemos 400 muestras del preámbulo (STS + GI2 + LTS + SIGNAL)
                                # para visualizar su estructura en el cuadrante Q3.
                                if sts_abs_idx + 400 <= len(bloque_iq):
                                    preamble_raw = bloque_iq[sts_abs_idx : sts_abs_idx + 400]
                                    envolvente_preambulo = np.abs(preamble_raw)
                                
                                break

            # 3. CÁLCULO DE ESPECTRO
            # Si encontramos un burst, usamos el chunk sincronizado (más limpio).
            # Si no, usamos el inicio del bloque para seguir mostrando algo.
            chunk_psd = chunk_trigger if chunk_trigger is not None else bloque_iq[:fs].copy()
            chunk_psd = chunk_psd - np.mean(chunk_psd)
            potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk_psd)))**2 / fs
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            
            # Interpolamos el bin DC para tapar el spike de hardware
            centro = fs // 2
            PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0
            
            resultados = {
                'psd_rf': PSD,
                'rf_chunk': chunk_trigger,
                'mpx_time': envolvente_preambulo,  # Reutilizamos la clave mpx_time para no tocar main.py
            }

            with self._lock:
                self.last_heavy_results = resultados
                self.nuevos_datos_listos = True
            
        finally:
            self.is_processing = False