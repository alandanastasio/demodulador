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
            
            # --- Paso 1: Sincronización Schmidl & Cox y CFO ---
            # --- Paso 1: Sincronización Schmidl & Cox y CFO ---
            Tu = self.Tu
            cp_len_default = self.cp_len_2
            
            # Correlación de prefijo cíclico
            prod = chunk[:-Tu] * np.conjugate(chunk[Tu:])
            window = np.ones(cp_len_default)
            from scipy import signal
            corr = signal.correlate(prod, window, mode='valid')
            
            power = np.abs(chunk[Tu:])**2
            power_smooth = signal.correlate(power, window, mode='valid')
            metric = np.abs(corr) / np.maximum(power_smooth, 1e-12)
            
            # Buscamos el pico en el primer ms
            limite_busqueda = int(self.sample_rate * 0.001)
            if limite_busqueda < len(metric):
                mejor_pos = np.argmax(metric[:limite_busqueda])
            else:
                mejor_pos = np.argmax(metric)
                
            # Estimación y corrección de CFO fraccional
            angulo = np.angle(corr[mejor_pos])
            cfo_fraccional_hz = angulo / (2 * np.pi * Tu / self.sample_rate)
            t_vec = np.arange(len(chunk))
            chunk_corregido = chunk * np.exp(-1j * 2 * np.pi * cfo_fraccional_hz * t_vec / self.sample_rate)
            
            self.ultimo_lte_metrics['cfo_hz'] = cfo_fraccional_hz
            self.ultimo_lte_metrics['cfo_fraccional_hz'] = cfo_fraccional_hz

            # --- Paso 2: Construcción Inicial (Pass 1) para hallar DMRS ---
            grid_freq = []
            idx = mejor_pos
            half_shift = np.exp(-1j * np.pi * np.arange(Tu) / Tu)
            
            for i in range(14):
                if idx + self.cp_len_1 + Tu > len(chunk_corregido):
                    break
                simbolo_t = chunk_corregido[idx + cp_len_default : idx + cp_len_default + Tu]
                simbolo_t_shift = simbolo_t * half_shift
                simbolo_f = np.fft.fftshift(np.fft.fft(simbolo_t_shift))
                simbolo_f_activo = simbolo_f[self.occupied_subcarriers]
                grid_freq.append(simbolo_f_activo)
                idx += cp_len_default + Tu
                
            grid_freq = np.array(grid_freq)
            
            # --- Paso 3: Identificación Ciega de DMRS ---
            dmrs_f = np.array([])
            const_L = np.array([])
            const_R = np.array([])
            
            if len(grid_freq) == 14:
                # Filtrar RBs inactivos (evitar procesar ruido)
                num_rbs = grid_freq.shape[1] // 12
                rb_power = np.array([np.mean(np.abs(grid_freq[:, r*12:(r+1)*12])**2) for r in range(num_rbs)])
                
                if len(rb_power) > 0:
                    max_pwr = np.max(rb_power)
                    peak_rb = np.argmax(rb_power)
                    
                    start_rb = peak_rb
                    while start_rb > 0 and rb_power[start_rb - 1] > 0.05 * max_pwr:
                        start_rb -= 1
                        
                    end_rb = peak_rb
                    while end_rb < num_rbs - 1 and rb_power[end_rb + 1] > 0.05 * max_pwr:
                        end_rb += 1
                        
                    active_subcarriers = np.arange(start_rb*12, (end_rb+1)*12)
                else:
                    active_subcarriers = np.arange(grid_freq.shape[1])
                    
                grid_freq_active = grid_freq[:, active_subcarriers]

                # DMRS tiene magnitud constante en frecuencia
                var_mags = []
                for i in range(14):
                    mag = np.abs(grid_freq_active[i])
                    mag_norm = mag / (np.mean(mag) + 1e-9)
                    var_mags.append(np.var(mag_norm))
                
                # Buscar par separado por 7
                mejor_par = 0
                min_var = float('inf')
                for i in range(7):
                    var_par = var_mags[i] + var_mags[i+7]
                    if var_par < min_var:
                        min_var = var_par
                        mejor_par = i
                        
                # PASS 2: Re-extracción con CP lengths correctos
                # Si mejor_par es el primer DMRS en nuestra ventana, sabemos que en la trama
                # de LTE el primer DMRS está en el símbolo 3 del slot. 
                # Por lo tanto, los símbolos 0 y 7 del slot (que tienen CP largo) están a
                # una distancia de -3 y +4 respecto al DMRS.
                grid_freq_exact = []
                idx = mejor_pos
                for k in range(14):
                    if idx + self.cp_len_1 + Tu > len(chunk_corregido):
                        break
                    es_sym0 = ((k - mejor_par) % 7 == 4)
                    cp = self.cp_len_1 if es_sym0 else self.cp_len_2
                    
                    simbolo_t = chunk_corregido[idx + cp : idx + cp + Tu]
                    simbolo_t_shift = simbolo_t * half_shift
                    simbolo_f = np.fft.fftshift(np.fft.fft(simbolo_t_shift))
                    simbolo_f_activo = simbolo_f[self.occupied_subcarriers[active_subcarriers]]
                    grid_freq_exact.append(simbolo_f_activo)
                    idx += cp + Tu
                    
                if len(grid_freq_exact) == 14:
                    grid_freq_exact = np.array(grid_freq_exact)
                    
                    dmrs_indices = [mejor_par, mejor_par + 7]
                    pusch_indices = [i for i in range(14) if i not in dmrs_indices]
                    
                    dmrs_f = grid_freq_exact[dmrs_indices].flatten()
                    pusch_f = grid_freq_exact[pusch_indices]
                    
                    # Ecualización de canal en frecuencia (solo amplitud, DMRS = Zadoff-Chu)
                    dmrs_matrix = grid_freq_exact[dmrs_indices]
                    dmrs_mag_avg = np.mean(np.abs(dmrs_matrix), axis=0)
                    
                    kernel_size = min(13, max(3, pusch_f.shape[1] // 20))
                    if kernel_size % 2 == 0: kernel_size += 1
                    kernel = np.ones(kernel_size) / kernel_size
                    dmrs_mag_smooth = np.convolve(dmrs_mag_avg, kernel, mode='same')
                    
                    target_amp = np.mean(dmrs_mag_smooth)
                    W_amp = target_amp / np.maximum(dmrs_mag_smooth, 1e-10)
                    pusch_f = pusch_f * W_amp[np.newaxis, :]
                    
                    # Ecualización Ciega (Fine Timing) con SC-FDMA
                    M = pusch_f.shape[1]
                    centro = self.fft_size // 2
                    # Índices físicos exactos para corrección de fase (importante si no está centrado)
                    k_phys = self.occupied_subcarriers[active_subcarriers] - centro
                    
                    best_dt = 0
                    min_cm_cost = float('inf')
                    pusch_t_best = None
                    
                    for dt in np.linspace(-5, 5, 21):
                        phase_ramp = np.exp(-1j * 2 * np.pi * k_phys * dt / Tu)
                        pusch_f_comp = pusch_f * phase_ramp
                        pusch_t_test = np.fft.ifft(pusch_f_comp, axis=1) * np.sqrt(M)
                        
                        cost = np.var(np.abs(pusch_t_test))
                        if cost < min_cm_cost:
                            min_cm_cost = cost
                            best_dt = dt
                            pusch_t_best = pusch_t_test
                    
                    # Seguimiento de Fase y CFO Residual Robusto (para QPSK y 16/64QAM)
                    # El CFO residual produce una rotación de fase lineal símbolo a símbolo.
                    # Extraemos la fase elevando a la 4ta potencia, la desenvolvemos y ajustamos una recta
                    # para filtrar drásticamente el ruido de patrón, permitiendo recuperar 16QAM.
                    phases_4 = []
                    for sym in pusch_t_best:
                        m4 = np.mean(sym**4)
                        phases_4.append(np.angle(m4))
                        
                    phases_4 = np.unwrap(phases_4)
                    
                    x = np.arange(len(pusch_t_best))
                    slope, intercept = np.polyfit(x, phases_4, 1)
                    
                    # Dividimos por 4 la fase reconstruida (slope y intercept de la 4ta potencia)
                    phase_corr = (slope * x + intercept) / 4.0
                    
                    pusch_t_eq = []
                    for i, sym in enumerate(pusch_t_best):
                        pusch_t_eq.append(sym * np.exp(-1j * phase_corr[i]))
                        
                    # Rotar 45 grados para alinear a los cuadrantes estándar (la 4ta potencia tiene un offset de pi)
                    pusch_t_eq = np.array(pusch_t_eq).flatten() * np.exp(1j * np.pi/4)
                        
                    const_L = pusch_t_eq.real
                    const_R = pusch_t_eq.imag
                    self.ultimo_lte_metrics['pss_found'] = True
                else:
                    self.ultimo_lte_metrics['pss_found'] = False
            else:
                self.ultimo_lte_metrics['pss_found'] = False

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
                'audio_time_L': const_L,
                'audio_time_R': const_R,
                'psd_mpx': np.array([]),
                'f_axis_mpx': np.array([]),
                'metricas': {
                    'lte_metrics': self.ultimo_lte_metrics,
                    'pss_pts': dmrs_f  # Guardamos DMRS aquí para que se pinte en naranja si se activa PSS
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
