import numpy as np
import threading
import time
from .base import DemoduladorBase

def generar_pss_time(fft_size: int):
    # Genera las 3 secuencias PSS en el dominio del tiempo (Zadoff-Chu)
    roots = [25, 29, 34] # N_ID_2 = 0, 1, 2
    pss_time = []
    
    for u in roots:
        # Secuencia de longitud 62
        n = np.arange(0, 31)
        d_u_1 = np.exp(-1j * np.pi * u * n * (n + 1) / 63)
        n2 = np.arange(31, 62)
        d_u_2 = np.exp(-1j * np.pi * u * (n2 + 1) * (n2 + 2) / 63)
        d_u = np.concatenate((d_u_1, d_u_2))
        
        # Mapeo a las 62 subportadoras centrales (DC vacío)
        X = np.zeros(fft_size, dtype=complex)
        X[fft_size-31 : fft_size] = d_u[0:31]
        X[1 : 32] = d_u[31:62]
        
        # IFFT para pasar a tiempo
        x = np.fft.ifft(X) * np.sqrt(fft_size)
        pss_time.append(x)
        
    return pss_time

class DemoduladorLTE(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 30.72e6 
        self.fft_size = 2048
        
        # Parámetros básicos de la trama LTE (ej. ancho de banda 20 MHz)
        self.Tu = 2048 # Tiempo útil del símbolo
        self.cp_len_1 = 160 # CP del primer símbolo del slot (normal CP)
        self.cp_len_2 = 144 # CP del resto de los símbolos del slot
        
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
        self.ultimo_puntos_corr = None

    @property
    def id(self): return "lte"

    @property
    def nombre_mostrar(self): return "LTE (OFDM/SC-FDMA)"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.ui_fft_size = fft_size
        
        # Mapeo de frecuencias de muestreo estandar LTE a tamaño de FFT
        # 1.92, 3.84, 7.68, 15.36, 23.04, 30.72 MHz
        fs_mhz = round(sample_rate / 1e6, 2)
        
        if fs_mhz <= 1.92:
            self.fft_size = 128
        elif fs_mhz <= 3.84:
            self.fft_size = 256
        elif fs_mhz <= 7.68:
            self.fft_size = 512
        elif fs_mhz <= 15.36:
            self.fft_size = 1024
        elif fs_mhz <= 23.04:
            self.fft_size = 1536
        else:
            self.fft_size = 2048
            
        self.Tu = self.fft_size
        # CP length in us: 5.2 for 1st symbol, 4.69 for the rest
        self.cp_len_1 = int(np.round(5.2e-6 * sample_rate))
        self.cp_len_2 = int(np.round(4.69e-6 * sample_rate))
        
        self.pss_time = generar_pss_time(self.fft_size)
        
        self.buffer_medicion = []
        self.muestras_acumuladas = 0
        self.is_processing = False
        
        with self._lock:
            self.nuevos_datos_listos = False
            self.last_heavy_results = {}
            self.ultimo_chunk_norm = None
            self.ultimo_lte_metrics = {}
            self.ultimo_puntos_corr = None

    def procesar(self, muestras_iq):
        with self._lock:
            if self.nuevos_datos_listos:
                self.nuevos_datos_listos = False
                return self.last_heavy_results

        ahora = time.time()
        
        if self.is_processing or ahora < self.proxima_captura:
            return None

        self.is_processing = True

        threading.Thread(
            target=self._procesar_fondo,
            args=(muestras_iq.copy(),),
            daemon=True
        ).start()

        return None

    def _procesar_fondo(self, bloque_iq: np.ndarray):
        try:
            fs = self.fft_size
            N_iq = len(bloque_iq)
            
            # --- FASE 1 & 2: Sincronización y Búsqueda de PSS (Downlink) ---
            mejor_corr = 0
            mejor_N_id_2 = -1
            mejor_pos = -1
            
            # Correlación cruzada rápida usando FFT (Circular)
            if N_iq >= fs:
                bloque_fft = np.fft.fft(bloque_iq)
                for n_id_2, pss_t in enumerate(self.pss_time):
                    pss_pad = np.zeros(N_iq, dtype=complex)
                    pss_pad[:fs] = pss_t
                    pss_fft = np.fft.fft(pss_pad)
                    
                    corr = np.fft.ifft(bloque_fft * np.conj(pss_fft))
                    corr_abs = np.abs(corr)
                    
                    max_val = np.max(corr_abs)
                    if max_val > mejor_corr:
                        mejor_corr = max_val
                        mejor_N_id_2 = n_id_2
                        mejor_pos = np.argmax(corr_abs)
                        
                self.ultimo_lte_metrics['pss_found'] = True
                self.ultimo_lte_metrics['N_id_2'] = mejor_N_id_2
                self.ultimo_lte_metrics['pss_pos'] = mejor_pos
            else:
                self.ultimo_lte_metrics['pss_found'] = False
            
            # --- FASE 3: Remoción de CP, FFT y Extracción de Subtrama ---
            subframe_fft = None
            if mejor_corr > 100: # Umbral empírico para considerar que hay señal real
                # El PSS en FDD está en el último símbolo (índice 6) del primer slot (slot 0)
                # Vamos a retroceder para encontrar el inicio de la subtrama (símbolo 0)
                muestras_atras = 5 * (fs + self.cp_len_2) + (fs + self.cp_len_1)
                inicio_trama = mejor_pos - muestras_atras
                
                # Un subframe tiene 14 símbolos (Normal CP)
                # 2 símbolos tipo 1 (índices 0 y 7) y 12 símbolos tipo 2
                longitud_subframe = 2 * (fs + self.cp_len_1) + 12 * (fs + self.cp_len_2)
                
                if inicio_trama >= 0 and inicio_trama + longitud_subframe <= N_iq:
                    subframe_fft = []
                    idx_actual = inicio_trama
                    
                    for num_sym in range(14):
                        es_sym0 = (num_sym % 7 == 0)
                        cp_len = self.cp_len_1 if es_sym0 else self.cp_len_2
                        
                        # Extraemos solo el tiempo útil (ignoramos CP)
                        idx_tu = idx_actual + cp_len
                        simbolo_tu = bloque_iq[idx_tu : idx_tu + fs]
                        
                        # Pasamos a frecuencia
                        simbolo_f = np.fft.fftshift(np.fft.fft(simbolo_tu)) / np.sqrt(fs)
                        subframe_fft.append(simbolo_f)
                        
                        idx_actual += cp_len + fs
                        
                    subframe_fft = np.array(subframe_fft)
                    self.ultimo_lte_metrics['trama_valida'] = True
                else:
                    self.ultimo_lte_metrics['trama_valida'] = False
            
            # --- FASE 4: Constelación y Métricas para UI ---
            evm_data = None
            puntos_corr = np.array([])
            
            if subframe_fft is not None:
                # Extraemos todas las subportadoras de datos (excluyendo márgenes y DC)
                # En un canal de, por ejemplo, 20 MHz, usamos 1200 subportadoras centrales
                num_sc = int((fs / 2048) * 1200) # Aproximación escalada según tabla
                if num_sc > fs - 2: num_sc = fs - 2
                
                centro = fs // 2
                mitad_sc = num_sc // 2
                
                # Índices de subportadoras (sin DC)
                idx_portadoras = list(range(centro - mitad_sc, centro)) + list(range(centro + 1, centro + mitad_sc + 1))
                
                constelacion = subframe_fft[:, idx_portadoras]
                
                # Simulamos ecualización burda (solo normalización de potencia)
                # (TODO: En la próxima iteración usaremos Cell-Specific Reference Signals para ecualizar fase)
                p_avg = np.mean(np.abs(constelacion)**2)
                if p_avg > 0:
                    constelacion = constelacion / np.sqrt(p_avg)
                
                puntos_corr = constelacion.flatten()
                
                # Armamos métricas EVM falsas para visualizar la estructura en UI
                # (Hasta no tener ecualización fina, el EVM será muy alto)
                evm_sym = np.random.uniform(-10, -5, 14) # Un EVM ruidoso temporal
                evm_subc = np.random.uniform(-15, -8, num_sc)
                eje_x_subc = np.concatenate((np.arange(-mitad_sc, 0), np.arange(1, mitad_sc + 1)))
                
                evm_data = {
                    'subc_x': eje_x_subc,
                    'subc_rms': evm_subc,
                    'subc_peak': evm_subc + 2,
                    'sym_rms': evm_sym,
                    'sym_peak': evm_sym + 3
                }
                
                self.ultimo_puntos_corr = puntos_corr

            # Generamos espectro visual para UI usando la resolución que pide el usuario
            ui_fs = getattr(self, 'ui_fft_size', fs)
            # Aseguramos tener suficientes muestras
            muestras_req = min(ui_fs, N_iq)
            chunk_psd = bloque_iq[:muestras_req].copy()
            if len(chunk_psd) < ui_fs:
                chunk_psd = np.pad(chunk_psd, (0, ui_fs - len(chunk_psd)))
                
            chunk_psd = chunk_psd - np.mean(chunk_psd)
            potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk_psd, n=ui_fs)))**2 / ui_fs
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            
            centro_psd = ui_fs // 2
            PSD[centro_psd] = (PSD[centro_psd - 1] + PSD[centro_psd + 1]) / 2.0
            
            resultados = {
                'psd_rf': PSD,
                'rf_chunk': np.abs(bloque_iq[:min(fs*2, N_iq)]), 
                'mpx_time': np.array([]),  
                'audio_time_L': puntos_corr.real if len(puntos_corr) > 0 else np.array([]),
                'audio_time_R': puntos_corr.imag if len(puntos_corr) > 0 else np.array([]),
                'psd_mpx': np.array([]),
                'f_axis_mpx': np.array([]),
                'metricas': {'lte_metrics': self.ultimo_lte_metrics},
                'evm_data': evm_data
            }

            with self._lock:
                self.last_heavy_results = resultados
                self.nuevos_datos_listos = True
            
        finally:
            self.is_processing = False
            self.proxima_captura = time.time() + self.pausa_entre_snapshots
