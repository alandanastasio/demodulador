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

def generar_sss(N_id_1: int, N_id_2: int, subframe: int = 0):
    q_prime = N_id_1 // 30
    q = (N_id_1 + (q_prime * (q_prime + 1)) // 2) // 30
    m_prime = N_id_1 + (q * (q + 1)) // 2
    
    m0 = m_prime % 31
    m1 = (m0 + (m_prime // 31) + 1) % 31
    
    def get_m_seq(poly_indices):
        x = np.zeros(31, dtype=int)
        x[4] = 1
        for i in range(26):
            val = 0
            for idx in poly_indices:
                val ^= x[i + idx]
            x[i + 5] = val % 2
        return 1 - 2 * x
        
    s_tilde = get_m_seq([2, 0])
    c_tilde = get_m_seq([3, 0])
    z_tilde = get_m_seq([4, 2, 1, 0])
    
    s0 = np.array([s_tilde[(n + m0) % 31] for n in range(31)])
    s1 = np.array([s_tilde[(n + m1) % 31] for n in range(31)])
    c0 = np.array([c_tilde[(n + N_id_2) % 31] for n in range(31)])
    c1 = np.array([c_tilde[(n + N_id_2 + 3) % 31] for n in range(31)])
    z1_m0 = np.array([z_tilde[(n + (m0 % 8)) % 31] for n in range(31)])
    z1_m1 = np.array([z_tilde[(n + (m1 % 8)) % 31] for n in range(31)])
    
    d = np.zeros(62)
    for n in range(31):
        if subframe == 0:
            d[2*n] = s0[n] * c0[n]
            d[2*n + 1] = s1[n] * c1[n] * z1_m0[n]
        else: # subframe 5
            d[2*n] = s1[n] * c0[n]
            d[2*n + 1] = s0[n] * c1[n] * z1_m1[n]
    return d

def generar_secuencia_gold(length, c_init):
    """Genera la secuencia pseudo-aleatoria c(n) según 3GPP TS 36.211 §7.2.
    Usa dos m-sequences x1 y x2 de largo 31, con Nc=1600 de offset."""
    Nc = 1600
    total = length + Nc
    
    x1 = np.zeros(total + 31, dtype=int)
    x2 = np.zeros(total + 31, dtype=int)
    
    # x1 se inicializa con x1(0)=1
    x1[0] = 1
    
    # x2 se inicializa con c_init en binario
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    
    # Generar las m-sequences
    for n in range(total):
        x1[n + 31] = (x1[n + 3] + x1[n]) % 2
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) % 2
    
    c = np.zeros(length, dtype=int)
    for n in range(length):
        c[n] = (x1[n + Nc] + x2[n + Nc]) % 2
    
    return c

def generar_crs(cell_id, ns, l, num_rb):
    """Genera los pilotos CRS para un slot ns, símbolo l, según 3GPP TS 36.211 §6.10.1.
    
    Args:
        cell_id: Physical Cell ID (N_cell_id)
        ns: Número de slot (0-19)
        l: Índice de símbolo OFDM dentro del slot (0 o 4 para normal CP)
        num_rb: Número de Resource Blocks del sistema (ej: 15 para 3 MHz)
    
    Returns:
        r_l: Secuencia compleja de pilotos CRS (2*num_rb valores)
    """
    N_maxRB = 110  # máximo RBs en LTE
    c_init = (1 << 10) * (7 * (ns + 1) + l + 1) * (2 * cell_id + 1) + 2 * cell_id + 1  # Ecuación estándar: adaptada para Normal CP (Ncp=1)
    
    c = generar_secuencia_gold(4 * N_maxRB, c_init)
    
    # r(m) = (1/sqrt(2)) * (1-2*c(2m)) + j*(1/sqrt(2)) * (1-2*c(2m+1))
    m = np.arange(2 * num_rb) + N_maxRB - num_rb
    r_l = (1/np.sqrt(2)) * (1 - 2*c[2*m].astype(float)) + \
          1j * (1/np.sqrt(2)) * (1 - 2*c[2*m + 1].astype(float))
    
    return r_l

def ecualizar_con_crs(subframe_fft, cell_id, fft_size, num_rb, ns_base=0):
    """Estima el canal usando los pilotos CRS y ecualiza todos los símbolos del subframe.
    
    CRS se ubican en los símbolos 0 y 4 de cada slot (normal CP, puerto 0).
    Dentro de cada símbolo, los pilotos van cada 6 subportadoras con offset = cell_id % 6.
    
    Args:
        subframe_fft: Array (14, fft_size) con los símbolos en frecuencia (fftshift aplicado)
        cell_id: Physical Cell ID
        fft_size: Tamaño de la FFT
        num_rb: Número de Resource Blocks
        ns_base: Número del primer slot absoluto del subframe (ej: 0 para sf0, 10 para sf5)
    
    Returns:
        subframe_eq: Array ecualizado (misma forma que subframe_fft)
    """
    centro = fft_size // 2
    num_sc = num_rb * 12  # subportadoras ocupadas totales
    mitad_sc = num_sc // 2
    
    # Índices absolutos de las subportadoras ocupadas (saltando DC)
    idx_portadoras = np.array(list(range(centro - mitad_sc, centro)) + 
                              list(range(centro + 1, centro + mitad_sc + 1)))
    
    v_shift = cell_id % 6
    
    # Los símbolos con CRS en normal CP, puerto 0: símbolo 0 y 4 de cada slot
    # En un subframe (2 slots): símbolos 0, 4, 7, 11
    # (l_in_slot, slot_absoluto)
    crs_symbols = [(0, ns_base), (4, ns_base), (0, ns_base + 1), (4, ns_base + 1)]
    sym_indices = [0, 4, 7, 11]  # índice global en el subframe
    
    # Para el offset vertical (v): símbolo 0 tiene v=0, símbolo 4 tiene v=3
    v_offsets = {0: 0, 4: 3}
    
    # Estimación de canal: H_est en cada subportadora
    H_est = np.ones((14, fft_size), dtype=complex)
    
    for sym_global, (l, ns) in zip(sym_indices, crs_symbols):
        crs_ref = generar_crs(cell_id, ns, l, num_rb)
        
        # Offset del piloto en la grilla de subportadoras
        v = v_offsets[l]
        pilot_offset = (v + v_shift) % 6
        
        # Posiciones de los pilotos dentro de las subportadoras ocupadas (0-indexed)
        pilot_local = np.arange(pilot_offset, num_sc, 6)
        
        # Posiciones absolutas en el array fftshift (saltando DC)
        pilot_abs = idx_portadoras[pilot_local]
        
        # Extraer los valores recibidos en las posiciones de piloto
        rx_pilots = subframe_fft[sym_global, pilot_abs]
        
        n_pilots = len(pilot_local)
        ref_pilots = crs_ref[:n_pilots]
        
        # Estimación LS: H = Rx / Ref
        H_pilots = rx_pilots / ref_pilots
        
        # Asignación nearest-neighbor: cada subportadora usa el piloto más cercano.
        pilot_positions = pilot_local
        all_positions = np.arange(num_sc)
        
        # Para cada subportadora, encontrar el piloto más cercano
        nearest_idx = np.argmin(np.abs(all_positions[:, None] - pilot_positions[None, :]), axis=1)
        H_interp = H_pilots[nearest_idx]
        
        H_est[sym_global, idx_portadoras] = H_interp
    
    # Interpolar H entre los 4 símbolos CRS para los símbolos intermedios.
    # Usamos nearest-neighbor también en el dominio temporal para evitar cancelaciones.
    sym_pos = np.array(sym_indices)
    all_sym = np.arange(14)
    
    for sc in idx_portadoras:
        h_values = H_est[sym_indices, sc]
        nearest_idx = np.argmin(np.abs(all_sym[:, None] - sym_pos[None, :]), axis=1)
        H_est[:, sc] = h_values[nearest_idx]
    
    # Ecualización ZF vectorizada: Y_eq = Y / H_est
    subframe_eq = subframe_fft.copy()
    mask = np.abs(H_est[:, idx_portadoras]) > 1e-10
    subframe_eq[:, idx_portadoras] = np.where(mask, subframe_fft[:, idx_portadoras] / H_est[:, idx_portadoras], subframe_fft[:, idx_portadoras])
    
    return subframe_eq

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
            self.buffer_medicion.extend(bloque_iq)
            muestras_10ms = int(self.sample_rate * 0.01)
            
            if len(self.buffer_medicion) < muestras_10ms:
                return # Esperamos a tener 10 ms de captura
                
            # Procesamos exactamente 10 ms
            chunk_procesar = np.array(self.buffer_medicion[:muestras_10ms])
            self.buffer_medicion = self.buffer_medicion[muestras_10ms:]
            
            fs = self.fft_size
            N_iq = len(chunk_procesar)
            
            # --- FASE 1 & 2: Sincronización y Búsqueda de PSS (Downlink) ---
            mejor_corr = 0
            mejor_N_id_2 = -1
            mejor_pos = -1
            
            bloque_fft = np.fft.fft(chunk_procesar)
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
                    
            # Validamos el pico de correlación contra el ruido de fondo (aprox 4x o 5x superior)
            es_pico_valido = mejor_corr > (4.0 * np.mean(corr_abs))
            
            if es_pico_valido:
                self.ultimo_lte_metrics['pss_found'] = True
                self.ultimo_lte_metrics['N_id_2'] = mejor_N_id_2
                self.ultimo_lte_metrics['pss_pos'] = mejor_pos
            else:
                self.ultimo_lte_metrics['pss_found'] = False
            
            # --- FASE 3: Remoción de CP, FFT y Extracción de Subtrama ---
            subframe_fft = None
            if es_pico_valido:
                # mejor_pos apunta al INICIO del símbolo OFDM del PSS (sin CP).
                # El PSS está en el símbolo 6 del subframe (último del slot 0).
                # Para ir del inicio del PSS al inicio del subframe:
                #   cp_del_PSS + 5*(Tu+cp_corto) + (Tu+cp_largo)
                muestras_atras = self.cp_len_2 + 5 * (fs + self.cp_len_2) + (fs + self.cp_len_1)
                inicio_trama = mejor_pos - muestras_atras
                
                longitud_subframe = 2 * (fs + self.cp_len_1) + 12 * (fs + self.cp_len_2)
                
                # Determinamos en qué mitad de la trama de 10ms cayó el PSS
                muestras_5ms = int(self.sample_rate * 0.005)
                pss_en_segunda_mitad = (mejor_pos >= muestras_5ms)
                
                # Si el PSS está muy al principio y nos caemos del arreglo,
                # usamos el segundo PSS de la trama (que está a 5ms exactos)
                if inicio_trama < 0:
                    inicio_trama += muestras_5ms
                    pss_en_segunda_mitad = True
                
                if inicio_trama >= 0 and inicio_trama + longitud_subframe <= N_iq:
                    subframe_fft = []
                    idx_actual = inicio_trama
                    
                    for num_sym in range(14):
                        es_sym0 = (num_sym % 7 == 0)
                        cp_len = self.cp_len_1 if es_sym0 else self.cp_len_2
                        
                        idx_tu = idx_actual + cp_len
                        simbolo_tu = chunk_procesar[idx_tu : idx_tu + fs]
                        
                        simbolo_f = np.fft.fftshift(np.fft.fft(simbolo_tu))
                        subframe_fft.append(simbolo_f)
                        
                        idx_actual += cp_len + fs
                        
                    subframe_fft = np.array(subframe_fft)
                    self.ultimo_lte_metrics['trama_valida'] = True
                    
                    # --- FASE 3.1: Decodificación SSS y Cell ID ---
                    # El SSS está en el símbolo 5 (justo antes del PSS en el símbolo 6)
                    sss_f = subframe_fft[5]
                    centro = fs // 2
                    idx_sss = list(range(centro - 31, centro)) + list(range(centro + 1, centro + 32))
                    sss_rx = sss_f[idx_sss]
                    
                    mejor_corr_sss = 0
                    mejor_N_id_1 = -1
                    
                    # El subframe depende de dónde cayó el PSS en la trama
                    subf_idx = 5 if pss_en_segunda_mitad else 0
                    
                    for n_id_1 in range(168):
                        d_ref = generar_sss(n_id_1, mejor_N_id_2, subf_idx)
                        corr = np.abs(np.vdot(d_ref, sss_rx))
                        if corr > mejor_corr_sss:
                            mejor_corr_sss = corr
                            mejor_N_id_1 = n_id_1
                            
                    cell_id = 3 * mejor_N_id_1 + mejor_N_id_2
                    self.ultimo_lte_metrics['N_id_1'] = mejor_N_id_1
                    self.ultimo_lte_metrics['cell_id'] = cell_id
                    
                    # --- FASE 5: Estimación de Canal (CRS) y Ecualización ---
                    fs_to_rb = {128: 6, 256: 15, 512: 25, 1024: 50, 1536: 75, 2048: 100}
                    num_rb = fs_to_rb.get(fs, 15)
                    # Los slots absolutos dependen del subframe:
                    # subframe 0 → slots 0,1 | subframe 5 → slots 10,11
                    ns_base = 10 if pss_en_segunda_mitad else 0
                    subframe_fft = ecualizar_con_crs(subframe_fft, cell_id, fs, num_rb, ns_base)
                    
                else:
                    self.ultimo_lte_metrics['trama_valida'] = False
            
            # --- FASE 4: Constelación y Métricas para UI ---
            evm_data = None
            puntos_corr = np.array([])
            
            # Generación de Envolvente Temporal Decimada para la UI (10 ms)
            num_puntos_ui = 2000
            if N_iq > num_puntos_ui:
                factor = N_iq // num_puntos_ui
                env_bruta = np.abs(chunk_procesar[:factor * num_puntos_ui])
                rf_chunk_ui = np.max(env_bruta.reshape(-1, factor), axis=1)
            else:
                rf_chunk_ui = np.abs(chunk_procesar)
            
            if subframe_fft is not None:
                # Mapeo exacto de tamaño de FFT a cantidad de subportadoras de datos (excluyendo márgenes y DC)
                fs_to_sc = {128: 72, 256: 180, 512: 300, 1024: 600, 1536: 900, 2048: 1200}
                num_sc = fs_to_sc.get(fs, int((fs / 2048) * 1200))
                
                centro = fs // 2
                mitad_sc = num_sc // 2
                
                idx_portadoras = list(range(centro - mitad_sc, centro)) + list(range(centro + 1, centro + mitad_sc + 1))
                constelacion = subframe_fft[:, idx_portadoras]
                
                p_avg = np.mean(np.abs(constelacion)**2)
                if p_avg > 0:
                    constelacion = constelacion / np.sqrt(p_avg)
                
                puntos_corr = constelacion.flatten()
                
                # --- EVM real por símbolo y por subportadora ---
                # Detectamos el punto ideal más cercano de la constelación QPSK
                qpsk_ref = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
                
                evm_por_sym = np.zeros(14)
                evm_por_subc = np.zeros(num_sc)
                
                for s in range(14):
                    sym_pts = constelacion[s, :]
                    # Para cada punto, encontrar el punto QPSK más cercano
                    distancias = np.abs(sym_pts[:, None] - qpsk_ref[None, :])  # (num_sc, 4)
                    idx_min = np.argmin(distancias, axis=1)
                    errores = sym_pts - qpsk_ref[idx_min]
                    evm_por_sym[s] = np.sqrt(np.mean(np.abs(errores)**2))
                    evm_por_subc += np.abs(errores)**2
                
                evm_por_subc = np.sqrt(evm_por_subc / 14)
                
                # Convertir a dB
                evm_sym_db = 20 * np.log10(np.maximum(evm_por_sym, 1e-10))
                evm_subc_db = 20 * np.log10(np.maximum(evm_por_subc, 1e-10))
                
                eje_x_subc = np.concatenate((np.arange(-mitad_sc, 0), np.arange(1, mitad_sc + 1)))
                
                evm_data = {
                    'subc_x': eje_x_subc,
                    'subc_rms': evm_subc_db,
                    'subc_peak': evm_subc_db + 2,
                    'sym_rms': evm_sym_db,
                    'sym_peak': evm_sym_db + 3
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
                'rf_chunk': rf_chunk_ui, 
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
        except Exception as e:
            print(f"Error en _procesar_fondo LTE: {e}")
        finally:
            self.is_processing = False
            self.proxima_captura = time.time() + self.pausa_entre_snapshots
