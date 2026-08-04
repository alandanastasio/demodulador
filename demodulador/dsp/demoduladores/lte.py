import numpy as np
import threading
import time
from .base import DemoduladorBase
from scipy import signal

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
    
    # Estimación de canal: H_est_0 (Puerto 0) y H_est_1 (Puerto 1)
    H_est_0 = np.ones((14, fft_size), dtype=complex)
    H_est_1 = np.ones((14, fft_size), dtype=complex)
    
    # Para el offset vertical (v):
    # Port 0: símbolo 0 tiene v=0, símbolo 4 tiene v=3
    # Port 1: símbolo 0 tiene v=3, símbolo 4 tiene v=0
    v_offsets_p0 = {0: 0, 4: 3}
    v_offsets_p1 = {0: 3, 4: 0}
    
    for sym_global, (l, ns) in zip(sym_indices, crs_symbols):
        crs_ref = generar_crs(cell_id, ns, l, num_rb)
        
        for port, v_offsets, H_est in [(0, v_offsets_p0, H_est_0), (1, v_offsets_p1, H_est_1)]:
            v = v_offsets[l]
            pilot_offset = (v + v_shift) % 6
            pilot_local = np.arange(pilot_offset, num_sc, 6)
            pilot_abs = idx_portadoras[pilot_local]
            
            rx_pilots = subframe_fft[sym_global, pilot_abs]
            n_pilots = len(pilot_local)
            ref_pilots = crs_ref[:n_pilots]
            
            H_pilots = rx_pilots / ref_pilots
            
            all_positions = np.arange(num_sc)
            H_real = np.interp(all_positions, pilot_local, H_pilots.real)
            H_imag = np.interp(all_positions, pilot_local, H_pilots.imag)
            H_interp = H_real + 1j * H_imag
            
            H_est[sym_global, idx_portadoras] = H_interp
            
    # Interpolación lineal en el tiempo para los símbolos intermedios
    sym_pos = np.array(sym_indices)
    all_sym = np.arange(14)
    
    for sc in idx_portadoras:
        for H_est in [H_est_0, H_est_1]:
            h_values = H_est[sym_indices, sc]
            h_real = np.interp(all_sym, sym_pos, h_values.real)
            h_imag = np.interp(all_sym, sym_pos, h_values.imag)
            H_est[:, sc] = h_real + 1j * h_imag
            
    # Ecualización MMSE vectorizada (SISO, Puerto 0): Y_eq = Y * H0* / (|H0|^2 + sigma2)
    subframe_eq = subframe_fft.copy()
    
    margen_guarda = 20
    idx_ruido = np.concatenate((np.arange(0, max(0, centro - mitad_sc - margen_guarda)), 
                                np.arange(min(fft_size, centro + mitad_sc + 1 + margen_guarda), fft_size)))
    
    if len(idx_ruido) > 0:
        sigma2 = np.var(subframe_fft[:, idx_ruido])
    else:
        sigma2 = 0.05
        
    Y = subframe_fft[:, idx_portadoras]
    H0 = H_est_0[:, idx_portadoras]
    
    Y_eq = (Y * np.conj(H0)) / (np.abs(H0)**2 + sigma2)
    subframe_eq[:, idx_portadoras] = Y_eq
    
    return subframe_eq, H_est_0, H_est_1, sigma2

def alamouti_combine(Y, H0, H1, sigma2):
    """Combina símbolos usando SFBC (Alamouti) para 2 antenas."""
    Y1 = Y[0::2]
    Y2 = Y[1::2]
    
    H0_1 = H0[0::2]
    H0_2 = H0[1::2]
    
    H1_1 = H1[0::2]
    H1_2 = H1[1::2]
    
    D = np.abs(H0_1)**2 + np.abs(H1_1)**2 + sigma2
    
    S1 = (np.conj(H0_1) * Y1 + H1_2 * np.conj(Y2)) / D
    S2 = (np.conj(H1_1) * Y1 - H0_2 * np.conj(Y2)) / D
    
    out = np.zeros_like(Y)
    out[0::2] = S1
    out[1::2] = S2
    return out

# --- DECODIFICADOR PBCH ---
def decodificar_pbch(soft_bits_480):
    # 1. Rate de-matching circular (480 -> 120)
    w = np.zeros(120)
    for i in range(480):
        w[i % 120] += soft_bits_480[i]
        
    # 2. De-interlace en 3 streams
    v0 = w[0::3]
    v1 = w[1::3]
    v2 = w[2::3]
    
    # 3. Sub-block de-interleaving
    valid_seq = np.array([8, 24, 16, 0, 32, 12, 28, 20, 4, 36, 10, 26, 18, 2, 34, 14, 30, 22, 6, 38, 9, 25, 17, 1, 33, 13, 29, 21, 5, 37, 11, 27, 19, 3, 35, 15, 31, 23, 7, 39])
    
    d0 = np.zeros(40)
    d1 = np.zeros(40)
    d2 = np.zeros(40)
    for k, original_idx in enumerate(valid_seq):
        d0[original_idx] = v0[k]
        d1[original_idx] = v1[k]
        d2[original_idx] = v2[k]
        
    soft_bits_3streams = np.vstack((d0, d1, d2)).T
    
    # 4. Decodificador Viterbi (TBCC K=7, Tasa 1/3)
    num_states = 64
    next_state = np.zeros((num_states, 2), dtype=int)
    outputs = np.zeros((num_states, 2, 3), dtype=int)
    
    for state in range(num_states):
        for bit in (0, 1):
            ns = (bit << 5) | (state >> 1)
            out0 = (bit ^ ((state >> 4)&1) ^ ((state >> 3)&1) ^ ((state >> 1)&1) ^ (state & 1))
            out1 = (bit ^ ((state >> 5)&1) ^ ((state >> 4)&1) ^ ((state >> 3)&1) ^ (state & 1))
            out2 = (bit ^ ((state >> 5)&1) ^ ((state >> 4)&1) ^ ((state >> 2)&1) ^ (state & 1))
            next_state[state, bit] = ns
            outputs[state, bit] = [out0, out1, out2]
            
    path_metrics = np.zeros(num_states)
    
    # Convergencia para el Tail-Biting
    for run in range(3):
        for i in range(40):
            new_metrics = np.full(num_states, -np.inf)
            for state in range(num_states):
                for bit in (0, 1):
                    ns = next_state[state, bit]
                    expected = 1 - 2 * outputs[state, bit]
                    branch_metric = np.sum(soft_bits_3streams[i] * expected)
                    if path_metrics[state] + branch_metric > new_metrics[ns]:
                        new_metrics[ns] = path_metrics[state] + branch_metric
            path_metrics = new_metrics
            
    # Traceback
    best_state = np.argmax(path_metrics)
    tb_states = np.zeros((40, num_states), dtype=int)
    tb_bits = np.zeros((40, num_states), dtype=int)
    
    path_metrics = np.full(num_states, -np.inf)
    path_metrics[best_state] = 0
    
    for i in range(40):
        new_metrics = np.full(num_states, -np.inf)
        new_tb_states = np.zeros(num_states, dtype=int)
        new_tb_bits = np.zeros(num_states, dtype=int)
        for state in range(num_states):
            if path_metrics[state] == -np.inf: continue
            for bit in (0, 1):
                ns = next_state[state, bit]
                expected = 1 - 2 * outputs[state, bit]
                branch_metric = np.sum(soft_bits_3streams[i] * expected)
                if path_metrics[state] + branch_metric > new_metrics[ns]:
                    new_metrics[ns] = path_metrics[state] + branch_metric
                    new_tb_states[ns] = state
                    new_tb_bits[ns] = bit
        path_metrics = new_metrics
        tb_states[i] = new_tb_states
        tb_bits[i] = new_tb_bits
        
    final_state = np.argmax(path_metrics)
    curr = final_state
    decoded = []
    for i in range(39, -1, -1):
        decoded.append(tb_bits[i, curr])
        curr = tb_states[i, curr]
        
    decoded = np.array(decoded[::-1])
    
    # 5. Verificación CRC16 y Antenas
    reg = 0
    for bit in decoded[:24]:
        msb = (reg >> 15) & 1
        reg = ((reg << 1) & 0xFFFF)
        if msb ^ bit:
            reg ^= 0x1021
            
    crc_recibido = 0
    for bit in decoded[24:]:
        crc_recibido = (crc_recibido << 1) | bit
        
    mask = reg ^ crc_recibido
    
    antenas = 0
    if mask == 0x0000: antenas = 1
    elif mask == 0xFFFF: antenas = 2
    elif mask == 0x5555: antenas = 4
    
    return decoded, antenas, mask

def decodificar_pcfich(simbolos_16, cell_id, n_s=0):
    # Demodulación QPSK a bits blandos (soft bits)
    # bit0 -> Real, bit1 -> Imag
    soft_bits = []
    for sym in simbolos_16:
        soft_bits.append(sym.real)
        soft_bits.append(sym.imag)
    soft_bits = np.array(soft_bits)
    
    # Scrambling
    c_init = (n_s // 2 + 1) * (2 * cell_id + 1) * 512 + cell_id
    c = generar_secuencia_gold(32, c_init)
    descrambled = soft_bits * (1 - 2*c)
    
    # Palabras código (Codewords) para CFI
    cfi_1 = np.array([0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1])
    cfi_2 = np.array([1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0])
    cfi_3 = np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1])
    
    # Correlación cruzada
    score_1 = np.sum((1 - 2*cfi_1) * descrambled)
    score_2 = np.sum((1 - 2*cfi_2) * descrambled)
    score_3 = np.sum((1 - 2*cfi_3) * descrambled)
    
    scores = [score_1, score_2, score_3]
    max_idx = np.argmax(scores)
    cfi = max_idx + 1
    
    # Validación: el score ganador debe superar al segundo por un margen claro
    sorted_scores = sorted(scores, reverse=True)
    margen = sorted_scores[0] - sorted_scores[1]
    es_valido = sorted_scores[0] > 3.0 and margen > 2.0
    
    return cfi, scores[max_idx], es_valido

class DemoduladorLTE(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 30.72e6 
        self.fft_size = 2048
        
        # Parámetros básicos de la trama LTE (ej. ancho de banda 20 MHz)
        self.Tu = 2048 # Tiempo útil del símbolo
        self.cp_len_1 = 160 # CP del primer símbolo del slot (normal CP)
        self.cp_len_2 = 144 # CP del resto de los símbolos del slot
        
        self.buffer_medicion = []
        self.is_processing = False
        
        self.last_heavy_results = {}
        self.nuevos_datos_listos = False
        self._lock = threading.Lock()
        self.pausa_entre_snapshots = 0.05
        self.proxima_captura = 0.0
        
        self.ultimo_chunk_norm = None
        self.ultimo_lte_metrics = {}
        self.ultimo_puntos_corr = np.array([])

    @property
    def id(self): return "lte"

    @property
    def nombre_mostrar(self): return "LTE (OFDM/SC-FDMA)"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
            
        self.Tu = self.fft_size
        # CP length in us: 5.2 for 1st symbol, 4.69 for the rest
        self.cp_len_1 = int(np.round(5.2e-6 * sample_rate))
        self.cp_len_2 = int(np.round(4.69e-6 * sample_rate))
        
        self.pss_time = generar_pss_time(self.fft_size)
        
        self.buffer_medicion = []
        self.is_processing = False
        
        with self._lock:
            self.nuevos_datos_listos = False
            self.last_heavy_results = {}
            self.ultimo_chunk_norm = None
            self.ultimo_lte_metrics = {}
            self.ultimo_puntos_corr = np.array([])

    def procesar(self, muestras_iq):
        with self._lock:
            if self.nuevos_datos_listos:
                self.nuevos_datos_listos = False
                return self.last_heavy_results

        if muestras_iq is None:
            self.buffer_medicion = []
            return None

        ahora = time.time()
        
        if self.is_processing or ahora < self.proxima_captura:
            return None

        self.buffer_medicion.extend(muestras_iq)
        muestras_10ms = int(self.sample_rate * 0.01)
        
        if len(self.buffer_medicion) < muestras_10ms:
            return None
            
        chunk_procesar = np.array(self.buffer_medicion[:muestras_10ms])
        self.buffer_medicion = []

        self.is_processing = True

        threading.Thread(
            target=self._procesar_fondo,
            args=(chunk_procesar,),
            daemon=True
        ).start()

        return None

    def _procesar_fondo(self, chunk_procesar: np.ndarray):
        try:
            fs = self.fft_size
            N_iq = len(chunk_procesar)
            
            # --- FASE 1: Corrección de CFO (Schmidl & Cox) ---
            original = chunk_procesar[:-fs]
            retraso = chunk_procesar[fs:]
            producto = retraso * np.conjugate(original)
            
            filtro = np.ones(self.cp_len_1)
            correlacion_sc = np.convolve(producto, filtro, mode='valid')
            
            inicio_simbolo = np.argmax(np.abs(correlacion_sc))
            pico_fase = np.angle(correlacion_sc[inicio_simbolo])
            
            ts = 1.0 / self.sample_rate
            cfo_estimado_hz = pico_fase / (2 * np.pi * fs * ts)
            
            t_vector = np.arange(N_iq) * ts
            chunk_procesar = chunk_procesar * np.exp(-1j * 2 * np.pi * cfo_estimado_hz * t_vector)

            # --- FASE 1.5 & 2: Sincronización y Búsqueda de PSS (Downlink) ---
            mejor_corr = 0
            mejor_N_id_2 = -1
            mejor_pos = -1
            mejor_corr_abs = None
            
            for n_id_2, pss_t in enumerate(self.pss_time):
                # Correlación lineal rápida usando superposición-suma (overlap-add/save)
                # Esto soluciona los bugs de "wrap-around" cíclico que daba np.fft en el código anterior.
                # 'valid' asegura que sólo calculamos posiciones donde la secuencia pss_t entra completa.
                corr = signal.correlate(chunk_procesar, pss_t, mode='valid', method='fft')
                corr_abs = np.abs(corr)
                
                max_val = np.max(corr_abs)
                if max_val > mejor_corr:
                    mejor_corr = max_val
                    mejor_N_id_2 = n_id_2
                    mejor_pos = np.argmax(corr_abs)
                    mejor_corr_abs = corr_abs
                    
            # Validamos el pico de correlación contra el ruido de fondo (aprox 4x o 5x superior)
            es_pico_valido = mejor_corr_abs is not None and mejor_corr > (4.0 * np.mean(mejor_corr_abs))
            
            if es_pico_valido:
                self.ultimo_lte_metrics['pss_found'] = True
                self.ultimo_lte_metrics['N_id_2'] = mejor_N_id_2
                self.ultimo_lte_metrics['pss_pos'] = mejor_pos
            else:
                self.ultimo_lte_metrics['pss_found'] = False
                self.ultimo_lte_metrics['trama_valida'] = False
                self.ultimo_lte_metrics['pbch_ok'] = False
                self.ultimo_lte_metrics['pcfich_ok'] = False
            
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
                
                # Si el PSS está muy al principio y nos caemos del arreglo,
                # usamos el segundo PSS de la trama (que está a 5ms exactos)
                if inicio_trama < 0:
                    inicio_trama += int(self.sample_rate * 0.005)
                
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
                    mejor_subf_idx = 0
                    
                    # Como el PSS es idéntico en subtrama 0 y 5, no sabemos cuál atrapamos.
                    # Probamos ambas hipótesis para el SSS.
                    for subf in (0, 5):
                        for n_id_1 in range(168):
                            d_ref = generar_sss(n_id_1, mejor_N_id_2, subf)
                            corr = np.abs(np.vdot(d_ref, sss_rx))
                            if corr > mejor_corr_sss:
                                mejor_corr_sss = corr
                                mejor_N_id_1 = n_id_1
                                mejor_subf_idx = subf
                                
                    pss_en_segunda_mitad = (mejor_subf_idx == 5)
                    
                    cell_id = 3 * mejor_N_id_1 + mejor_N_id_2
                    self.ultimo_lte_metrics['N_id_1'] = mejor_N_id_1
                    self.ultimo_lte_metrics['cell_id'] = cell_id
                    
                    # --- FASE 5: Estimación de Canal (CRS) y Ecualización ---
                    fs_to_rb = {128: 6, 256: 15, 512: 25, 1024: 50, 1536: 75, 2048: 100}
                    num_rb = fs_to_rb.get(fs, 15)
                    # Los slots absolutos dependen del subframe:
                    # subframe 0 → slots 0,1 | subframe 5 → slots 10,11
                    ns_base = 10 if pss_en_segunda_mitad else 0
                    
                    # Guardamos una copia del subframe crudo (raw) para el futuro decodificador SFBC de 2 antenas
                    subframe_fft_raw = subframe_fft.copy()
                    subframe_eq, H_est_0, H_est_1, sigma2 = ecualizar_con_crs(subframe_fft, cell_id, fs, num_rb, ns_base)
                    
                    # Por ahora, sobreescribimos subframe_fft con la versión ecualizada SISO 
                    # para no romper la compatibilidad con el resto del pipeline hasta el Paso 3.
                    subframe_fft = subframe_eq
                    
                    # --- FASE 6: Extracción y Desaleatorización del PBCH ---
                    # El PBCH se encuentra sólo en el subframe 0, símbolos 7, 8, 9 y 10.
                    # Ocupa las 72 subportadoras centrales (sin incluir DC), esquivando los CRS.
                    if not pss_en_segunda_mitad:  # Sólo en subframe 0
                        idx_pbch = list(range(centro - 36, centro)) + list(range(centro + 1, centro + 37))
                        v_shift = cell_id % 6
                        pbch_siso = []
                        pbch_raw = []
                        pbch_h0 = []
                        pbch_h1 = []
                        
                        for l_slot in range(4):
                            sym_idx = 7 + l_slot
                            for k_idx, sc_abs in enumerate(idx_pbch):
                                k_local = k_idx + 54  # Para que el rango de 72 esté centrado en el bloque de 180 (90 - 36 = 54)
                                
                                is_crs = False
                                # El PBCH asume siempre 4 puertos de antena para el mapeo:
                                if l_slot == 0:
                                    if (k_local - (0 + v_shift)) % 6 == 0: is_crs = True # Puerto 0
                                    if (k_local - (3 + v_shift)) % 6 == 0: is_crs = True # Puerto 1
                                elif l_slot == 1:
                                    if (k_local - (0 + v_shift)) % 6 == 0: is_crs = True # Puerto 2
                                    if (k_local - (3 + v_shift)) % 6 == 0: is_crs = True # Puerto 3
                                    
                                if not is_crs:
                                    pbch_siso.append(subframe_fft[sym_idx, sc_abs])
                                    pbch_raw.append(subframe_fft_raw[sym_idx, sc_abs])
                                    pbch_h0.append(H_est_0[sym_idx, sc_abs])
                                    pbch_h1.append(H_est_1[sym_idx, sc_abs])
                                    
                        pbch_siso = np.array(pbch_siso)
                        pbch_raw = np.array(pbch_raw)
                        pbch_h0 = np.array(pbch_h0)
                        pbch_h1 = np.array(pbch_h1)
                        
                        if len(pbch_siso) == 240:
                            pbch_sfbc = alamouti_combine(pbch_raw, pbch_h0, pbch_h1, sigma2)
                            
                            c = generar_secuencia_gold(1920, cell_id)
                            fase_encontrada = False
                            antenas_detectadas = 0
                            mejor_pbch_eq = pbch_siso # Por defecto guardamos el SISO
                            
                            # Decodificación a ciegas: Probamos 1 antena (SISO) y luego 2 antenas (SFBC)
                            for mode, pbch_eq in [("SISO", pbch_siso), ("SFBC", pbch_sfbc)]:
                                soft_bits = np.empty(480)
                                soft_bits[0::2] = pbch_eq.real
                                soft_bits[1::2] = pbch_eq.imag
                                
                                for fase in range(4):
                                    c_fase = c[fase*480 : (fase+1)*480]
                                    scrambled = soft_bits * (1 - 2*c_fase)
                                    decoded_bits, antenas, mask = decodificar_pbch(scrambled)
                                    
                                    if (mode == "SISO" and antenas == 1) or (mode == "SFBC" and antenas == 2):
                                        fase_encontrada = True
                                        antenas_detectadas = antenas
                                        mejor_pbch_eq = pbch_eq
                                        break
                                if fase_encontrada:
                                    break
                                    
                            pbch_qpsk = mejor_pbch_eq  # Para usarlo luego en métricas (Fase 4)
                            
                            if antenas_detectadas > 0:
                                # ¡Match de CRC exitoso!
                                bits_24 = decoded_bits[:24]
                                mib_bits = "".join(map(str, bits_24))
                                
                                bw_val = (bits_24[0]<<2) | (bits_24[1]<<1) | bits_24[2]
                                bw_map = {0: '1.4 MHz', 1: '3 MHz', 2: '5 MHz', 3: '10 MHz', 4: '15 MHz', 5: '20 MHz'}
                                dl_bw = bw_map.get(bw_val, f"Desconocido ({bw_val})")
                                
                                phich_dur = "Normal" if bits_24[3] == 0 else "Extendido"
                                
                                phich_res_val = (bits_24[4]<<1) | bits_24[5]
                                res_map = {0: '1/6', 1: '1/2', 2: '1', 3: '2'}
                                phich_res = res_map.get(phich_res_val, str(phich_res_val))
                                
                                sfn_val = 0
                                for i in range(8):
                                    sfn_val = (sfn_val << 1) | bits_24[6+i]
                                
                                self.ultimo_lte_metrics['pbch_mib'] = mib_bits
                                self.ultimo_lte_metrics['pbch_antenas'] = antenas_detectadas
                                self.ultimo_lte_metrics['pbch_ok'] = True
                                self.ultimo_lte_metrics['mib_bw'] = dl_bw
                                self.ultimo_lte_metrics['mib_phich_dur'] = phich_dur
                                self.ultimo_lte_metrics['mib_phich_res'] = phich_res
                                self.ultimo_lte_metrics['mib_sfn'] = sfn_val
                                    
                                fase_encontrada = True
                            
                            if not fase_encontrada:
                                self.ultimo_lte_metrics['pbch_ok'] = False
                        else:
                            self.ultimo_lte_metrics['pbch_ok'] = False
                    
                    # --- DECODIFICACIÓN PCFICH ---
                    # El PCFICH existe en TODAS las subtramas, no sólo en la 0
                    sym0 = subframe_fft[0]
                    v_mod3 = (cell_id % 6) % 3
                    mitad_bw = num_rb * 6  # Mitad del ancho de banda en subportadoras
                    idx_portadoras = list(range(centro - mitad_bw, centro)) + list(range(centro + 1, centro + mitad_bw + 1))
                    
                    regs_k = []
                    current_reg_k = []
                    for i, sc in enumerate(idx_portadoras):
                        if (i % 3) != v_mod3:
                            current_reg_k.append(sc)
                            if len(current_reg_k) == 4:
                                regs_k.append(current_reg_k)
                                current_reg_k = []
                                
                    regs_k = np.array(regs_k)
                    n_reg = len(regs_k)
                    k_bar_reg = cell_id % n_reg
                    step_reg = n_reg // 4  # floor(N_REG / 4), equivalente a floor(N_RB_DL / 2)
                    reg_indices = [(k_bar_reg + i * step_reg) % n_reg for i in range(4)]
                    
                    pcfich_k_flat = np.concatenate([regs_k[idx] for idx in reg_indices])
                    self.ultimo_pcfich_k_indices = set(pcfich_k_flat)
                    
                    if self.ultimo_lte_metrics.get('tx_antennas', 1) == 2:
                        pcfich_syms = alamouti_combine(subframe_fft_raw[0, pcfich_k_flat], H_est_0[0, pcfich_k_flat], H_est_1[0, pcfich_k_flat], sigma2)
                        subframe_fft[0, pcfich_k_flat] = pcfich_syms
                    else:
                        pcfich_syms = subframe_fft[0, pcfich_k_flat]
                    
                    # --- EXTRACCIÓN APROXIMADA DE PHICH ---
                    regs_disponibles = [r for r in range(n_reg) if r not in reg_indices]
                    if len(regs_disponibles) >= 3:
                        step_phich = len(regs_disponibles) // 3
                        phich_reg_indices = [regs_disponibles[(cell_id + i * step_phich) % len(regs_disponibles)] for i in range(3)]
                        phich_k_flat = np.concatenate([regs_k[idx] for idx in phich_reg_indices])
                        self.ultimo_phich_k_indices = set(phich_k_flat)
                        
                        if self.ultimo_lte_metrics.get('tx_antennas', 1) == 2:
                            phich_syms = alamouti_combine(subframe_fft_raw[0, phich_k_flat], H_est_0[0, phich_k_flat], H_est_1[0, phich_k_flat], sigma2)
                            subframe_fft[0, phich_k_flat] = phich_syms
                        else:
                            phich_syms = subframe_fft[0, phich_k_flat]
                    else:
                        phich_syms = np.array([])
                        self.ultimo_phich_k_indices = set()
                    
                    # n_s es el número de slot (0 para subframe 0, 10 para subframe 5)
                    pcfich_n_s = ns_base
                    cfi_val, cfi_score, cfi_ok = decodificar_pcfich(pcfich_syms, cell_id, n_s=pcfich_n_s)
                    self.ultimo_lte_metrics['pcfich_cfi'] = cfi_val
                    self.ultimo_lte_metrics['pcfich_ok'] = cfi_ok
                    
                else:
                    self.ultimo_lte_metrics['trama_valida'] = False
                    
            if self.ultimo_lte_metrics.get('tx_antennas', 1) == 2 and subframe_fft is not None and self.ultimo_lte_metrics.get('trama_valida', True):
                # Arreglar la constelación de PBCH (que ya fue decodificada) reinyectándola para la UI
                if not pss_en_segunda_mitad and 'pbch_qpsk' in locals():
                    k_local_idx = 0
                    for l_slot in range(4):
                        sym_idx = 7 + l_slot
                        for k_idx, sc_abs in enumerate(idx_pbch):
                            k_local = k_idx + 54
                            is_crs = False
                            if l_slot == 0 and ((k_local - (0 + v_shift)) % 6 == 0 or (k_local - (3 + v_shift)) % 6 == 0): is_crs = True
                            if l_slot == 1 and ((k_local - (0 + v_shift)) % 6 == 0 or (k_local - (3 + v_shift)) % 6 == 0): is_crs = True
                            
                            if not is_crs:
                                idx_1d = l_slot * 60 + k_local_idx
                                if idx_1d < len(pbch_qpsk):
                                    subframe_fft[sym_idx, sc_abs] = pbch_qpsk[idx_1d]
                                k_local_idx += 1
                                
                # APLICACIÓN DE SFBC AL PDCCH
                tx_antennas = self.ultimo_lte_metrics.get('tx_antennas', 1)
                for sym_idx in range(cfi_val):
                    pdcch_k_abs = []
                    for k_local, k_abs in enumerate(idx_portadoras):
                        is_crs_port0 = sym_idx in [0, 4] and (k_local % 6) == v_shift
                        is_crs_port1 = sym_idx in [0, 4] and (k_local % 6) == (v_shift + 3) % 6
                        is_crs = is_crs_port0 or (tx_antennas > 1 and is_crs_port1)
                        
                        is_pcfich = (sym_idx == 0) and (k_abs in pcfich_k_flat)
                        is_phich = (sym_idx == 0) and (k_abs in phich_k_flat)
                        
                        if not (is_crs or is_pcfich or is_phich):
                            pdcch_k_abs.append(k_abs)
                            
                    pdcch_k_abs = np.array(pdcch_k_abs)
                    if len(pdcch_k_abs) > 0 and len(pdcch_k_abs) % 2 == 0:
                        pdcch_syms = alamouti_combine(
                            subframe_fft_raw[sym_idx, pdcch_k_abs], 
                            H_est_0[sym_idx, pdcch_k_abs], 
                            H_est_1[sym_idx, pdcch_k_abs], 
                            sigma2
                        )
                        subframe_fft[sym_idx, pdcch_k_abs] = pdcch_syms
            
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
            
            trama_valida = self.ultimo_lte_metrics.get('trama_valida', False)
            
            if subframe_fft is not None and trama_valida:
                # Mapeo exacto de tamaño de FFT a cantidad de subportadoras de datos (excluyendo márgenes y DC)
                fs_to_sc = {128: 72, 256: 180, 512: 300, 1024: 600, 1536: 900, 2048: 1200}
                num_sc = fs_to_sc.get(fs, int((fs / 2048) * 1200))
                
                centro = fs // 2
                mitad_sc = num_sc // 2
                
                idx_portadoras = list(range(centro - mitad_sc, centro)) + list(range(centro + 1, centro + mitad_sc + 1))
                constelacion = subframe_fft[:, idx_portadoras]
                
                idx_sync = np.array(list(range(mitad_sc - 31, mitad_sc)) + list(range(mitad_sc, mitad_sc + 31)))
                pss_pts = constelacion[6, idx_sync].copy()
                sss_pts = constelacion[5, idx_sync].copy()
                
                # El subframe ya fue ecualizado usando CRS en la FASE 5
                # por lo que no es necesario volver a normalizar la amplitud.
                
                cfi_val = self.ultimo_lte_metrics.get('pcfich_cfi', 1)
                if not (1 <= cfi_val <= 3): cfi_val = 1
                
                v_shift = self.ultimo_lte_metrics.get('cell_id', 0) % 6
                pcfich_k_indices = set()
                if hasattr(self, 'ultimo_pcfich_k_indices'):
                    pcfich_k_indices = self.ultimo_pcfich_k_indices
                
                phich_k_indices = set()
                if hasattr(self, 'ultimo_phich_k_indices'):
                    phich_k_indices = self.ultimo_phich_k_indices
                
                # Convertir índices absolutos (posición en FFT) a relativos (posición en constelacion)
                abs_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(idx_portadoras)}
                pcfich_k_rel = set(abs_to_rel[ki] for ki in pcfich_k_indices if ki in abs_to_rel)
                phich_k_rel = set(abs_to_rel[ki] for ki in phich_k_indices if ki in abs_to_rel)
                
                idx_pbch = np.array(list(range(mitad_sc - 36, mitad_sc)) + list(range(mitad_sc, mitad_sc + 36)))
                
                pdcch_pts = []
                crs_pts = []
                pbch_pts = []
                pcfich_pts = []
                phich_pts = []
                pdsch_pts = []
                
                tx_antennas = self.ultimo_lte_metrics.get('tx_antennas', 1)
                
                for sym_idx in range(14):
                    for k in range(num_sc):
                        pt = constelacion[sym_idx, k]
                        if np.isnan(pt): continue
                        
                        # Detectar C-RS
                        is_crs_port0 = sym_idx in [0, 4, 7, 11] and (k % 6) == v_shift
                        is_crs_port1 = sym_idx in [0, 4, 7, 11] and (k % 6) == (v_shift + 3) % 6
                        
                        if is_crs_port0 or (tx_antennas > 1 and is_crs_port1):
                            crs_pts.append(pt)
                            continue
                            
                        # PSS / SSS ya extraídos
                        if (sym_idx == 6 or sym_idx == 5) and (k in idx_sync):
                            continue
                            
                        # PBCH
                        if sym_idx in [7, 8, 9, 10] and (k in idx_pbch):
                            pbch_pts.append(pt)
                            continue
                            
                        # PCFICH
                        is_pcfich = (sym_idx == 0) and (k in pcfich_k_rel)
                        if is_pcfich:
                            pcfich_pts.append(pt)
                            continue
                            
                        # PHICH
                        is_phich = (sym_idx == 0) and (k in phich_k_rel)
                        if is_phich:
                            phich_pts.append(pt)
                            continue
                            
                        # PDCCH (Resto de la Región de control)
                        if sym_idx < cfi_val:
                            if np.abs(pt) > 0.1: # Ignorar Resource Elements vacíos
                                pdcch_pts.append(pt)
                            continue
                            
                        # Lo que sobra es PDSCH
                        if np.abs(pt) > 0.1:
                            pdsch_pts.append(pt)
                            
                pdcch_pts = np.array(pdcch_pts)
                crs_pts = np.array(crs_pts)
                pbch_pts = np.array(pbch_pts)
                pcfich_pts = np.array(pcfich_pts)
                phich_pts = np.array(phich_pts)
                pdsch_pts = np.array(pdsch_pts)
                
                # Guardar para UI (sin el límite de 0.1 para que se vea completo si quieren)
                puntos_corr = pdsch_pts
                self.ultimo_pss_pts = pss_pts
                self.ultimo_sss_pts = sss_pts
                
                def calc_evm_power(pts, ref_type='qpsk'):
                    if len(pts) == 0: return 0.0, 0.0
                    rms = np.sqrt(np.mean(np.abs(pts)**2) + 1e-12)
                    power = 10 * np.log10(rms**2)
                    
                    if ref_type == 'zchu':
                        err = np.abs(pts) - 1.0
                        evm = np.sqrt(np.mean(err**2)) * 100
                        return evm, power
                        
                    pts_norm = pts / rms
                    
                    if ref_type == 'qpsk':
                        ref = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
                    elif ref_type == 'bpsk':
                        ref = np.array([1, -1])
                    elif ref_type == '16qam':
                        a = np.array([-3, -1, 1, 3])
                        ref = np.array([x + 1j*y for x in a for y in a]) / np.sqrt(10)
                    elif ref_type == '64qam':
                        a = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
                        ref = np.array([x + 1j*y for x in a for y in a]) / np.sqrt(42)
                    elif ref_type == '256qam':
                        a = np.array([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
                        ref = np.array([x + 1j*y for x in a for y in a]) / np.sqrt(170)
                    
                    dist = np.abs(pts_norm[:, None] - ref[None, :])
                    idx = np.argmin(dist, axis=1)
                    err = pts_norm - ref[idx]
                    evm = np.sqrt(np.mean(np.abs(err)**2)) * 100
                    return evm, power
                    
                def format_evm(evm, power):
                    if evm == 0.0 and power == 0.0: return "---", "---"
                    return f"{evm:.2f}", f"{power:.2f}"
                
                # frame_summary guardará: { "Canal": (EVM_str, Power_str, NumRB_str) }
                fs_dict = {}
                fs_dict["P-SS"] = (*format_evm(*calc_evm_power(pss_pts, 'zchu')), "6")
                fs_dict["S-SS"] = (*format_evm(*calc_evm_power(sss_pts, 'bpsk')), "6")
                fs_dict["PBCH"] = (*format_evm(*calc_evm_power(pbch_pts, 'qpsk')), "6")
                fs_dict["C-RS"] = (*format_evm(*calc_evm_power(crs_pts, 'qpsk')), str(num_sc//12))
                fs_dict["PCFICH"] = (*format_evm(*calc_evm_power(pcfich_pts, 'qpsk')), "4")
                fs_dict["PHICH"] = (*format_evm(*calc_evm_power(phich_pts, 'bpsk')), "3")
                fs_dict["PDCCH"] = (*format_evm(*calc_evm_power(pdcch_pts, 'qpsk')), str(len(pdcch_pts)//12))
                
                # Detección Ciega de Modulación para PDSCH
                modulaciones = ['qpsk', '16qam', '64qam', '256qam']
                mejor_mod = 'qpsk'
                mejor_evm = 999.0
                mejor_pow = 0.0
                
                if len(pdsch_pts) > 0:
                    for mod in modulaciones:
                        evm_val, pow_val = calc_evm_power(pdsch_pts, mod)
                        if evm_val < mejor_evm:
                            mejor_evm = evm_val
                            mejor_pow = pow_val
                            mejor_mod = mod
                
                fs_dict[f"PDSCH_{mejor_mod.upper()}"] = (*format_evm(mejor_evm, mejor_pow), str(len(pdsch_pts)//12//11))
                self.ultimo_lte_metrics['pdsch_modulation'] = mejor_mod
                
                self.ultimo_lte_metrics['frame_summary'] = fs_dict

                # --- EVM real por símbolo y por portadora ---
                # Borrado a pedido del usuario (cálculos heredados de Wi-Fi no aplicables directamente a LTE)
                evm_data = {
                    'subc_x': [],
                    'subc_rms': [],
                    'subc_peak': [],
                    'sym_rms': [],
                    'sym_peak': []
                }

                
                self.ultimo_evm_data = evm_data
                self.ultimo_pbch_pts = pbch_pts
                self.ultimo_crs_pts = crs_pts
                self.ultimo_pcfich_pts = pcfich_pts
                self.ultimo_phich_pts = phich_pts
                self.ultimo_pdcch_pts = pdcch_pts
                
            else:
                puntos_corr = getattr(self, 'ultimo_puntos_corr', np.array([]))
                if puntos_corr is None:
                    puntos_corr = np.array([])
                evm_data = getattr(self, 'ultimo_evm_data', None)
                
            self.ultimo_puntos_corr = puntos_corr
            
            # Generamos espectro visual para UI usando la resolución de la FFT del canal
            ui_fs = fs
            # Aseguramos tener suficientes muestras
            muestras_req = min(ui_fs, N_iq)
            chunk_psd = chunk_procesar[:muestras_req].copy()
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
                'metricas': {
                    'lte_metrics': self.ultimo_lte_metrics,
                    'pss_pts': getattr(self, 'ultimo_pss_pts', np.array([])),
                    'sss_pts': getattr(self, 'ultimo_sss_pts', np.array([])),
                    'pdcch_pts': getattr(self, 'ultimo_pdcch_pts', np.array([])),
                    'pcfich_pts': getattr(self, 'ultimo_pcfich_pts', np.array([])),
                    'phich_pts': getattr(self, 'ultimo_phich_pts', np.array([])),
                    'pbch_pts': getattr(self, 'ultimo_pbch_pts', np.array([])),
                    'crs_pts': getattr(self, 'ultimo_crs_pts', np.array([]))
                },
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
