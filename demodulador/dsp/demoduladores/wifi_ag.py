import numpy as np
import threading
from scipy.ndimage import uniform_filter1d, binary_closing
from .base import DemoduladorBase
import time

# Constantes del preámbulo 802.11a/g (en muestras a 20 MHz)
_SC_N = 16   # Desplazamiento de la correlación S&C (mitad del STS period = 16 muestras)
_SC_W = 16   # Ventana de integración S&C

# --- SCHMIDL & COX ---
def schmidl_cox_metric(iq_signal, N=_SC_N, W=_SC_W):
    L = len(iq_signal)
    
    # 1. Productos cruzados y energía (directamente sobre la señal cruda)
    prod = np.conj(iq_signal[:-N]) * iq_signal[N:]
    energy = np.abs(iq_signal[N:]) ** 2
    
    # 2. Integración
    ventana = np.ones(W)
    P = np.convolve(prod, ventana, mode='valid')
    R = np.convolve(energy, ventana, mode='valid')
    
    # 3. Recorte
    P = P[:L - 2 * N]
    R = R[:L - 2 * N]
    
    # 4. Métrica final
    M = np.abs(P) ** 2 / (R ** 2 + 1e-10)
    
    return M, P, R

# Decodificador Viterbi rate 1/2
# Polinomios generadores: g0=133, g1=171 (octal) = 0b1011011, 0b1111001
# Constraint length K=7, memoria=6

def viterbi_decode(bits, K=7, g0=0b1011011, g1=0b1111001):
    """
    Decodificador Viterbi para codigo convolucional rate 1/2.
    Entrada: bits entrelazados como pares [b0, b1, b0, b1, ...]
    Salida: bits de informacion decodificados
    """
    n_states = 2 ** (K - 1)  # 64 estados
    INF = float('inf')

    # Precomputar salidas para cada estado y bit de entrada
    def conv_output(state, inp):
        reg = (inp << (K-1)) | state
        b0 = bin(reg & g0).count('1') % 2
        b1 = bin(reg & g1).count('1') % 2
        next_state = (inp << (K-2)) | (state >> 1)
        return next_state, b0, b1

    # Inicializar
    n_pairs = len(bits) // 2
    metrics = np.full(n_states, INF)
    metrics[0] = 0
    paths = np.zeros((n_pairs, n_states), dtype=int)
    prev_states = np.zeros((n_pairs, n_states), dtype=int)

    for t in range(n_pairs):
        rx0 = bits[2*t]
        rx1 = bits[2*t + 1]
        new_metrics = np.full(n_states, INF)

        for state in range(n_states):
            if metrics[state] == INF:
                continue
            for inp in [0, 1]:
                next_s, b0, b1 = conv_output(state, inp)
                # Distancia de Hamming
                dist = (b0 ^ rx0) + (b1 ^ rx1)
                m = metrics[state] + dist
                if m < new_metrics[next_s]:
                    new_metrics[next_s] = m
                    paths[t, next_s] = inp
                    prev_states[t, next_s] = state

        metrics = new_metrics

    # Traceback desde el estado con menor metrica
    decoded = np.zeros(n_pairs, dtype=int)
    state = np.argmin(metrics)
    for t in range(n_pairs - 1, -1, -1):
        decoded[t] = paths[t, state]
        state = prev_states[t, state]

    return decoded

def deinterleave_signal(bits, NCBPS=48, NBPSC=1):
    """
    Desentrelazador RX según IEEE 802.11-2007 §17.3.5.6.
    
    TX interleaver aplica dos permutaciones sobre los bits codificados:
      1ª permutación (k→i): i = (NCBPS/16)*(k mod 16) + floor(k/16)
      2ª permutación (i→j): j = s*floor(i/s) + (i + NCBPS - floor(16*i/NCBPS)) mod s
    
    RX debe invertir en orden inverso: primero deshacer la 2ª, luego la 1ª.
    """
    s = max(NBPSC // 2, 1)
    
    # --- Invertir la 2ª permutación (j → i) ---
    # Construimos el mapa forward i→j y lo invertimos
    fwd2 = np.zeros(NCBPS, dtype=int)
    for i in range(NCBPS):
        j = (s * (i // s) + (i + NCBPS - int(16 * i / NCBPS)) % s) % NCBPS
        fwd2[i] = j
    inv2 = np.zeros(NCBPS, dtype=int)
    for i in range(NCBPS):
        inv2[fwd2[i]] = i
    bits_step1 = bits[inv2]
    
    # --- Invertir la 1ª permutación (i → k) ---
    # Forward: i = (NCBPS/16)*(k mod 16) + floor(k/16)
    # Invertimos: coded[k] = step1[fwd1[k]]
    result = np.zeros(NCBPS, dtype=int)
    for k in range(NCBPS):
        i = (NCBPS // 16) * (k % 16) + k // 16
        result[k] = bits_step1[i]
    
    return result

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
        self.pausa_entre_snapshots = 0.05
        self.proxima_captura = 0.0
        self.ultimo_puntos_corr = None
        self.ultimo_wifi_metrics = {}
        self.ultimo_evm_data = None
        self.ultimo_S_data = None
        self.ultimo_chunk_norm = None
        self.ultimo_M_norm = None

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
            self.ultimo_puntos_corr = None
            self.ultimo_wifi_metrics = {}
            self.ultimo_evm_data = None
            self.ultimo_S_data = None
            self.ultimo_chunk_norm = None
            self.ultimo_M_norm = None

    def procesar(self, muestras_iq):
        if muestras_iq is None:
            self.buffer_medicion = []
            return None
        with self._lock:
            if self.nuevos_datos_listos:
                self.nuevos_datos_listos = False
                return self.last_heavy_results

        ahora = time.time()
        
        # Si estamos procesando o en pausa, descartamos este súper-bloque entero
        if self.is_processing or ahora < self.proxima_captura:
            return None

        # Si llegamos acá, muestras_iq YA ES el bloque de 4ms entero.
        self.is_processing = True

        threading.Thread(
            target=self._procesar_fondo,
            args=(muestras_iq.copy(),), # Le pasamos el paquete directo
            daemon=True
        ).start()

        return None

    def _procesar_fondo(self, bloque_iq: np.ndarray):
        try:
            fs = self.fft_size
            puntos_corr = None
            M_norm = np.array([])
            chunk_norm = np.array([])
            inicio_recorte = 0
            wifi_metrics = {}
            S_data = None

            # --- 0. LIMPIEZA DE HARDWARE ---
            # Eliminamos la fuga del oscilador local (DC Offset) de todo el bloque
            #bloque_iq = bloque_iq - np.mean(bloque_iq)

            # 1. BÚSQUEDA GRUESA (Energía)
            energia = np.abs(bloque_iq) ** 2
            energia_suave = uniform_filter1d(energia, size=50)
            max_energia = np.max(energia_suave)
            
            chunk_trigger = None
            envolvente_preambulo = None  # |preámbulo| para visualizar estructura STS/LTS en Q3
            wifi_metrics = {}
            

            energia_norm = energia_suave / max_energia
            en_burst_raw = energia_norm > 0.3
            
            # --- PROTECCIÓN CONTRA FALSO FIN DE BURST ---
            # Aplicamos cierre morfológico: Si hay una caída de energía menor a 100 muestras (5us)
            # producida por fading o ruido, se "rellena" conectando el burst.
            en_burst = binary_closing(en_burst_raw, structure=np.ones(100))
            
            cambios = np.diff(en_burst.astype(int))
            inicios_burst = np.where(cambios == 1)[0]
            fines_burst   = np.where(cambios == -1)[0]

            # Si la senal empieza ya dentro de un burst, agregar inicio en 0
            if en_burst[0]:
                inicios_burst = np.concatenate(([0], inicios_burst))

            n_bursts = min(len(inicios_burst), len(fines_burst))
            inicios_burst = inicios_burst[:n_bursts]
            fines_burst   = fines_burst[:n_bursts]

            # ---  RECORTE DEL BURST (chunk_norm) ---
            margen_muestras = int(10e-6 * self.sample_rate) # 10 us de margen (200 muestras a 20MHz)
            chunk_norm = energia_norm # Por defecto (si no hay bursts) mandamos todo

            inicio_recorte = 0
            
            if n_bursts >= 2:
                # Agarramos el segundo burst (índice 1)
                inicio_recorte = max(0, inicios_burst[1] - margen_muestras)
                fin_recorte = min(len(energia_norm), fines_burst[1] + margen_muestras)
                chunk_norm = energia_norm[inicio_recorte:fin_recorte]
            elif n_bursts == 1:
                # Fallback: Si solo detectó 1 burst, agarramos el primero (índice 0)
                inicio_recorte = max(0, inicios_burst[0] - margen_muestras)
                fin_recorte = min(len(energia_norm), fines_burst[0] + margen_muestras)
                chunk_norm = energia_norm[inicio_recorte:fin_recorte]

            if n_bursts >= 2:
                for i, (ini, fin) in enumerate(zip(inicios_burst[1:], fines_burst[1:]), start=1):
                    ini_ext = max(0, ini - int(0.5e-6 * self.sample_rate))
                    segmento = bloque_iq[ini_ext:fin]
                    
                    if len(segmento) < (_SC_N + _SC_W):
                        continue
                        
                    # 2. BÚSQUEDA FINA - Schmidl & Cox
                    M, P, R = schmidl_cox_metric(segmento)
                    if len(M) == 0:
                        continue
                    M_norm = M / np.max(M)

                    # Buscamos el primer índice donde la correlación normalizada supera 0.7
                    indices_sts = np.where(M_norm > 0.7)[0]
                    
                    # Validación de Meseta (Plateau Check): 
                    # El STS real dura ~160 muestras, la correlación debe mantenerse alta. 
                    if len(indices_sts) > 32:
                        # --- DETECTOR DE CAÍDA DE MESETA (Al estilo del max_counter de VHDL) ---
                        # En vez de usar la subida de la meseta (que depende de transitorios),
                        # buscamos exactamente dónde se termina la meseta (el falling edge).
                        saltos = np.where(np.diff(indices_sts) > 1)[0]
                        if len(saltos) > 0:
                            fin_meseta = indices_sts[saltos[0]]
                        else:
                            fin_meseta = indices_sts[-1]
                            
                        # La métrica empieza a caer teóricamente en la muestra 128 del STS, 
                        # y cruza el 0.7 aproximadamente en la muestra 132.
                        # Retrocedemos 132 muestras para encontrar el inicio exacto del STS.
                        muestra_local = fin_meseta - 132
                        
                        if muestra_local < 0:
                            muestra_local = indices_sts[0] # Fallback por si el bloque se recortó muy justo
                        muestra_abs = ini_ext + muestra_local
                        
                        margen_visual = 150
                        inicio_visual = max(0, muestra_abs - margen_visual)
                        
                        if inicio_visual + fs <= len(bloque_iq):
                            chunk_trigger = bloque_iq[inicio_visual : inicio_visual + fs].copy()
                            frame = bloque_iq[muestra_abs : fin]

                            # CFO estimation using STS
                            # La STS tiene 10 simbolos cortos de N=16 muestras
                            # El angulo de P(d) en la meseta es proporcional al CFO

                            # Tomar P en la zona de la meseta (donde M_norm > umbral)
                            P_meseta = P[indices_sts]
                            # CFO normalizado (en radianes por muestra)
                            cfo_rad = np.angle(np.mean(P_meseta)) / _SC_N

                            cfo_hz = cfo_rad * self.sample_rate / (2 * np.pi)
                            wifi_metrics['cfo'] = cfo_hz

                            # Validar que tengamos al menos las muestras del STS (160)
                            if len(frame) < 160:
                                continue

                            # Correccion del CFO en el frame completo
                            t_frame = np.arange(len(frame)) / self.sample_rate
                            frame_corr = frame * np.exp(-1j * 2 * np.pi * cfo_hz * t_frame)

                            # Extraer los 10 simbolos cortos del frame corregido (primeras 160 muestras)
                            sts = frame_corr[:10 * _SC_N].reshape(10, _SC_N)

                            # Promedio coherente = estimacion de la señal
                            s_ref = np.mean(sts, axis=0)

                            # Ruido = diferencia entre cada simbolo y la referencia
                            ruido = sts - s_ref
                            P_senal = np.mean(np.abs(s_ref) ** 2)
                            P_ruido  = np.mean(np.abs(ruido) ** 2)

                            snr_lineal = P_senal / P_ruido
                            snr_db     = 10 * np.log10(snr_lineal)
                            wifi_metrics['snr'] = snr_db

                            P_frame = np.mean(np.abs(frame_corr) ** 2)
                            gain_agc = 1.0 / np.sqrt(P_frame)
                            frame_norm = frame_corr * gain_agc

                            # Extraemos 400 muestras del preámbulo (STS + GI2 + LTS + SIGNAL)
                            # para visualizar su estructura en el cuadrante Q3.
                            if len(frame_norm) >= 400:
                                envolvente_preambulo = np.abs(frame_norm[:400])
                            
                            # LTS correcto según IEEE 802.11-2007, Ecuación (17-3)
                            # Orden de bins FFT: bin 0=DC, bin 1=+1, ..., bin 26=+26,
                            # bins 27-37=guard, bin 38=-26, ..., bin 63=-1
                            LTS_FREQ = np.array([
                            #    DC   +1   +2   +3   +4   +5   +6   +7   +8   +9  +10  +11  +12  +13  +14  +15
                                 0,   1,  -1,  -1,   1,   1,  -1,   1,  -1,   1,  -1,  -1,  -1,  -1,  -1,   1,
                            #  +16  +17  +18  +19  +20  +21  +22  +23  +24  +25  +26  guard...
                                 1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,   0,   0,   0,   0,   0,
                            #  guard...                                            -26  -25  -24  -23  -22  -21
                                 0,   0,   0,   0,   0,   0,   1,   1,  -1,  -1,   1,   1,
                            #  -20  -19  -18  -17  -16  -15  -14  -13  -12  -11  -10   -9   -8   -7   -6   -5
                                -1,   1,  -1,   1,   1,   1,   1,   1,   1,  -1,  -1,   1,   1,  -1,   1,  -1,
                            #   -4   -3   -2   -1
                                 1,   1,   1,   1
                            ], dtype=complex)

                            # Extraer el LTS del frame normalizado
                            N_STS = 10 * 16
                            N_GI2 = 32
                            N_LTS = 64

                            lts_raw = frame_norm[N_STS + N_GI2 : N_STS + N_GI2 + 2 * N_LTS]
                            lts1 = lts_raw[:N_LTS]
                            lts2 = lts_raw[N_LTS:]

                            LTS1_rx = np.fft.fft(lts1, N_LTS)
                            LTS2_rx = np.fft.fft(lts2, N_LTS)

                            # CFO fino sobre subportadoras activas
                            activas_lts = np.where(LTS_FREQ != 0)[0]
                            diff_fase = LTS2_rx[activas_lts] * np.conj(LTS1_rx[activas_lts])
                            cfo_fino_rad = np.angle(np.mean(diff_fase)) / N_LTS
                            cfo_fino_hz  = cfo_fino_rad * self.sample_rate / (2 * np.pi)
                            wifi_metrics['cfo_fino'] = cfo_fino_hz

                            # Aplicar CFO fino
                            t_frame2 = np.arange(len(frame_norm)) / self.sample_rate
                            frame_norm = frame_norm * np.exp(-1j * 2 * np.pi * cfo_fino_hz * t_frame2)

                            # Re-extraer LTS con frame corregido
                            lts_raw = frame_norm[N_STS + N_GI2 : N_STS + N_GI2 + 2 * N_LTS]
                            lts1 = lts_raw[:N_LTS]
                            lts2 = lts_raw[N_LTS:]

                            # Estimacion de canal: evitar division por cero
                            LTS1_rx = np.fft.fft(lts1, N_LTS)
                            LTS2_rx = np.fft.fft(lts2, N_LTS)
                            LTS_rx  = (LTS1_rx + LTS2_rx) / 2
                            H = np.zeros(N_LTS, dtype=complex)
                            H[activas_lts] = LTS_rx[activas_lts] / LTS_FREQ[activas_lts]

                            # Demodulacion del campo SIGNAL
                            # Ubicacion: justo despues de STS + GI2 + 2*LTS
                            N_CP_SIGNAL = 16   # prefijo ciclico
                            N_FFT       = 64

                            inicio_signal = N_STS + N_GI2 + 2 * N_LTS
                            signal_sym = frame_norm[inicio_signal + N_CP_SIGNAL : inicio_signal + N_CP_SIGNAL + N_FFT]

                            # FFT y ecualizacion
                            S = np.fft.fft(signal_sym, N_FFT)
                            S_eq = np.zeros(N_FFT, dtype=complex)
                            S_eq[activas_lts] = S[activas_lts] / H[activas_lts]

                            data_idx  = list(range(38, 64)) + list(range(1, 27))
                            pilot_idx = [43, 57, 7, 21]
                            data_idx  = [i for i in data_idx if i not in pilot_idx]

                            S_data = S_eq[data_idx]

                            # El campo SIGNAL usa BPSK: decidir por signo de la parte real
                            # Con el LTS corregido, la convención es: real>0 → bit 1, real<0 → bit 0
                            bits_raw = (S_data.real > 0).astype(int)
                            # Desentrelazado del campo SIGNAL (BPSK, NCBPS=48, NBPSC=1)
                            # Segun IEEE 802.11-2007 seccion 17.3.5.6

                            NCBPS = 48   # bits por simbolo OFDM para BPSK
                            NBPSC = 1    # bits por subportadora para BPSK
                            s = max(NBPSC // 2, 1)  # s=1 para BPSK

                            # Primera permutacion inversa: i -> k
                            # k = (NCBPS/16) * (i % 16) + floor(i/16)
                            bits_deint = deinterleave_signal(bits_raw)
                            bits_decoded = viterbi_decode(bits_deint)

                            # Parseo del campo SIGNAL
                            # bits 0-3:  RATE
                            # bit  4:    reservado
                            # bits 5-16: LENGTH (12 bits, LSB primero)
                            # bit  17:   paridad
                            # bits 18-23: tail (zeros)

                            # Parseo del campo SIGNAL
                            info_bits = bits_decoded[:18]

                            # RATE (bits 0-3, MSB primero)
                            rate_bits = info_bits[0:4]
                            rate_code = rate_bits[0]*8 + rate_bits[1]*4 + rate_bits[2]*2 + rate_bits[3]

                            rate_table = {
                                0b1101: ("BPSK",   "1/2",  6),
                                0b1111: ("BPSK",   "3/4",  9),
                                0b0101: ("QPSK",   "1/2", 12),
                                0b0111: ("QPSK",   "3/4", 18),
                                0b1001: ("16-QAM", "1/2", 24),
                                0b1011: ("16-QAM", "3/4", 36),
                                0b0001: ("64-QAM", "2/3", 48),
                                0b0011: ("64-QAM", "3/4", 54),
                            }

                            mod, code_rate, mbps = rate_table.get(rate_code, ("?", "?", 0))

                            # LENGTH (bits 5-16, LSB primero)
                            length_bits = info_bits[5:17]
                            length = sum(b << i for i, b in enumerate(length_bits))

                            # Paridad (bit 17): paridad par sobre bits 0-16
                            paridad_calc = np.sum(info_bits[0:17]) % 2
                            paridad_rx   = info_bits[17]
                            paridad_ok   = (paridad_calc == paridad_rx)

                            # TAIL (bits 18-23): Deben ser obligatoriamente ceros
                            tail_bits = bits_decoded[18:24]
                            tail_ok = (np.sum(tail_bits) == 0)

                            wifi_metrics.update({
                                'rate_code': bin(rate_code),
                                'mod': mod,
                                'code_rate': code_rate,
                                'mbps': mbps,
                                'length': length,
                                'paridad_ok': paridad_ok,
                                'tail_ok': tail_ok
                            })

                            # --- VALIDACIÓN ESTRICTA L-SIG (Rechazo de Falsos Positivos) ---
                            # Si la modulación no existe, la paridad falla, o el tail no es cero, NO es un paquete válido.
                            if mod == "?" or not paridad_ok or not tail_ok:
                                S_data = None
                                continue

                            # Demodulacion de los simbolos de datos (64-QAM)
                            N_CP  = 16
                            N_FFT = 64

                            # Subportadoras de datos (48) y pilotos (4)
                            data_idx  = list(range(38, 64)) + list(range(1, 27))
                            pilot_idx = [43, 57, 7, 21]
                            data_idx  = [i for i in data_idx if i not in pilot_idx]

                            # Inicio de los simbolos de datos: STS + GI2 + 2*LTS + SIGNAL
                            inicio_datos = N_STS + N_GI2 + 2 * N_LTS + (N_CP + N_FFT)

                            # --- CÁLCULO EXACTO DE LONGITUD L-SIG (Evasión de truncamiento) ---
                            N_DBPS = int(4 * mbps) # Bits de datos por símbolo OFDM
                            N_simbolos_exacto = int(np.ceil((16 + 8 * length + 6) / N_DBPS))

                            # Limitamos solo por si el hardware cortó el bloque físicamente
                            muestras_disponibles = len(frame_norm) - inicio_datos
                            N_simbolos_max = muestras_disponibles // (N_CP + N_FFT)
                            
                            N_simbolos = min(N_simbolos_exacto, N_simbolos_max)

                            # Demodular cada simbolo
                            constelacion = []
                            for k in range(N_simbolos):
                                offset = inicio_datos + k * (N_CP + N_FFT)
                                simbolo = frame_norm[offset + N_CP : offset + N_CP + N_FFT]
                                S = np.fft.fft(simbolo, N_FFT)
                                # Ecualizar
                                S_eq = np.zeros(N_FFT, dtype=complex)
                                S_eq[data_idx] = S[data_idx] / H[data_idx]
                                constelacion.append(S_eq[data_idx])

                            constelacion = np.array(constelacion)
                            puntos = constelacion.flatten()

                            # Correccion de fase simbolo a simbolo usando pilotos
                            # Pilotos en indices FFT: 7, 21, 43, 57 (+7, +21, -21, -7)
                            # Valores de referencia: [+1, +1, +1, -1] * secuencia_pn

                            # Secuencia PN de los pilotos (127 bits, polinomio x^7+x^4+1)
                            def pilot_pn_sequence(length):
                                reg = np.ones(7, dtype=int)
                                seq = []
                                for _ in range(length):
                                    seq.append(reg[6])
                                    feedback = reg[6] ^ reg[3]
                                    reg = np.roll(reg, 1)
                                    reg[0] = feedback
                                return np.array(seq)

                            pn = pilot_pn_sequence(N_simbolos + 1)
                            # Valor del piloto: 1 - 2*pn (mapeo 0->+1, 1->-1)
                            # Según IEEE 802.11, P_{-21, -7, 7, 21} = {1, 1, 1, -1}
                            # En orden de FFT (+7, +21, -21, -7) esto es [1, -1, 1, 1]
                            pilot_ref = np.array([1, -1, 1, 1])  # subportadoras +7,+21,-21,-7

                            pilot_idx_ordered = [7, 21, 43, 57]  # orden en FFT

                            constelacion_corr = []
                            pilots_corr = []
                            pilots_ideales = []
                            for k in range(N_simbolos):
                                offset = inicio_datos + k * (N_CP + N_FFT)
                                simbolo = frame_norm[offset + N_CP : offset + N_CP + N_FFT]
                                S = np.fft.fft(simbolo, N_FFT)
                                S_eq = S.copy()

                                # Ecualizar datos
                                S_eq[data_idx] = S[data_idx] / H[data_idx]

                                # Estimar fase residual con los pilotos
                                pn_k = 1 - 2 * pn[k]  # signo comun para este simbolo
                                pilots_rx  = S[pilot_idx_ordered] / H[pilot_idx_ordered]
                                pilots_exp = pilot_ref * pn_k
                                rot = pilots_rx * np.conj(pilots_exp)
                                fase_residual = np.angle(np.mean(rot))

                                # Corregir fase en las subportadoras de datos y pilotos
                                S_eq[data_idx] *= np.exp(-1j * fase_residual)
                                pilots_rx_corr = pilots_rx * np.exp(-1j * fase_residual)

                                constelacion_corr.append(S_eq[data_idx])
                                pilots_corr.append(pilots_rx_corr)
                                pilots_ideales.append(pilots_exp)

                            constelacion_corr = np.array(constelacion_corr)
                            puntos_corr = constelacion_corr.flatten()
                            pilots_corr = np.array(pilots_corr)
                            pilots_ideales = np.array(pilots_ideales)

                            # --- CÁLCULO DE EVM ---
                            H_datos = H[data_idx]
                            mask_validas = np.abs(H_datos) > 1e-6
                            
                            NIVELES_MODULACION = {
                                'BPSK':   np.array([-1, 1]),
                                'QPSK':   np.array([-1, 1]),
                                '16-QAM': np.array([-3, -1, 1, 3]),
                                '64-QAM': np.array([-7, -5, -3, -1, 1, 3, 5, 7])
                            }
                            
                            niveles_base = NIVELES_MODULACION.get(mod, np.array([-1, 1]))
                            multiplicador_2d = 1 if mod == 'BPSK' else 2
                            P_teorica = np.mean(niveles_base**2) * multiplicador_2d
                            
                            puntos_validos = constelacion_corr[:, mask_validas]
                            
                            if len(puntos_validos) > 0:
                                P_rx = np.mean(np.abs(puntos_validos)**2)
                                escala = np.sqrt(P_rx / P_teorica)
                                niveles_norm = niveles_base * escala
                                
                                def decisor_gen(puntos_1d, niveles, es_bpsk):
                                    I_dec = niveles[np.argmin(np.abs(puntos_1d.real[:,None] - niveles), axis=1)]
                                    if es_bpsk:
                                        Q_dec = np.zeros_like(I_dec)
                                    else:
                                        Q_dec = niveles[np.argmin(np.abs(puntos_1d.imag[:,None] - niveles), axis=1)]
                                    return I_dec + 1j * Q_dec

                                ideales_matrix = np.zeros_like(puntos_validos)
                                for k in range(N_simbolos):
                                    ideales_matrix[k] = decisor_gen(puntos_validos[k], niveles_norm, mod == 'BPSK')
                                    
                                P_ref = np.mean(np.abs(ideales_matrix)**2)
                                if P_ref == 0: P_ref = 1e-10
                                
                                evm_matrix_pct = np.abs(puntos_validos - ideales_matrix) / np.sqrt(P_ref) * 100
                                evm_pilots_pct = np.abs(pilots_corr - pilots_ideales) / np.sqrt(P_ref) * 100
                                
                                # Combinar datos y pilotos para EVM global (por símbolo)
                                evm_all_pct = np.concatenate((evm_matrix_pct, evm_pilots_pct), axis=1)
                                
                                # Por simbolo
                                evm_rms_sym  = np.sqrt(np.mean(evm_all_pct**2, axis=1))
                                evm_peak_sym = np.max(evm_all_pct, axis=1)
                                
                                # Por portadora (datos y pilotos)
                                evm_rms_subc_data = np.sqrt(np.mean(evm_matrix_pct**2, axis=0))
                                evm_peak_subc_data = np.max(evm_matrix_pct, axis=0)
                                evm_rms_subc_pilots = np.sqrt(np.mean(evm_pilots_pct**2, axis=0))
                                evm_peak_subc_pilots = np.max(evm_pilots_pct, axis=0)
                                
                                evm_rms_subc = np.concatenate((evm_rms_subc_data, evm_rms_subc_pilots))
                                evm_peak_subc = np.concatenate((evm_peak_subc_data, evm_peak_subc_pilots))
                                
                                def to_db(pct): return 20 * np.log10(np.maximum(pct, 1e-10) / 100)
                                
                                evm_rms_sym_db = to_db(evm_rms_sym)
                                evm_peak_sym_db = to_db(evm_peak_sym)
                                
                                # Reordenar subportadoras de -26 a +26
                                data_idx_validos = np.array(data_idx)[mask_validas]
                                # Concatenar indices de datos y pilotos
                                all_idx_validos = np.concatenate((data_idx_validos, pilot_idx_ordered))
                                subp_num = np.where(all_idx_validos < 32, all_idx_validos, all_idx_validos - 64)
                                orden = np.argsort(subp_num)
                                
                                evm_rms_subc_db = to_db(evm_rms_subc[orden])
                                evm_peak_subc_db = to_db(evm_peak_subc[orden])
                                subp_ordenadas = subp_num[orden]
                                
                                evm_data = {
                                    'subc_x': subp_ordenadas,
                                    'subc_rms': evm_rms_subc_db,
                                    'subc_peak': evm_peak_subc_db,
                                    'sym_rms': evm_rms_sym_db,
                                    'sym_peak': evm_peak_sym_db
                                }
                                
                                self.ultimo_puntos_corr = puntos_corr
                                self.ultimo_wifi_metrics = wifi_metrics
                                self.ultimo_evm_data = evm_data
                                self.ultimo_S_data = S_data
                                self.ultimo_chunk_norm = chunk_norm
                                self.ultimo_M_norm = M_norm

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
            
            # ENVIAMOS LA ÚLTIMA CONSTELACIÓN VÁLIDA Y BURST (FREEZE) PARA QUE NO TITILE
            p_corr = self.ultimo_puntos_corr
            s_dat = self.ultimo_S_data
            
            audio_L_out = p_corr.real if p_corr is not None else np.array([])
            audio_R_out = p_corr.imag if p_corr is not None else np.array([])
            
            resultados = {
                'psd_rf': PSD,
                'rf_chunk': self.ultimo_chunk_norm if self.ultimo_chunk_norm is not None else chunk_norm,
                'mpx_time': self.ultimo_M_norm if self.ultimo_M_norm is not None else M_norm,  
                'audio_time_L': audio_L_out,
                'audio_time_R': audio_R_out,
                'psd_mpx': s_dat.real if s_dat is not None else np.array([]),
                'f_axis_mpx': s_dat.imag if s_dat is not None else np.array([]),
                'metricas': {'inicio_recorte': inicio_recorte, 'wifi_metrics': self.ultimo_wifi_metrics},
                'evm_data': self.ultimo_evm_data
            }

            with self._lock:
                self.last_heavy_results = resultados
                self.nuevos_datos_listos = True
            
        finally:
            self.is_processing = False
            self.proxima_captura = (
                time.time() +
                self.pausa_entre_snapshots
            )