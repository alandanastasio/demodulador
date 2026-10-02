import numpy as np
import threading
import logging
import time
from collections import deque
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d
from .base import DemoduladorBase

# Constantes del preámbulo 802.11a/g (en muestras a 20 MHz)
_SC_N = 16   # Desplazamiento de la correlación S&C (mitad del STS period = 16 muestras)
_SC_W = 16   # Ventana de integración S&C

# Secuencia larga de entrenamiento 802.11a/g, en orden de bins FFT.
LTS_FREQ = np.array([
     0,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1,
     1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1,  1,  1,  1,
], dtype=np.complex128)
LTS_TIME = np.fft.ifft(LTS_FREQ)


def find_lts_start(segment, cfo_hz, sample_rate, plateau_start, plateau_end):
    """Ubica el primer LTS correlacionando los dos LTS consecutivos.

    Los bordes STS sólo delimitan la búsqueda; el máximo LTS fija el tiempo.
    Devuelve el índice del primer LTS dentro de ``segment`` o None.
    """
    # Un STS ocupa 160 muestras y GI2 otras 32. La meseta S&C suele
    # terminar ~30 muestras antes del final del STS.
    lo = max(0, plateau_start + 135, plateau_end + 20)
    hi = min(len(segment) - 128, plateau_end + 120)
    if hi < lo:
        return None

    search_end = hi + 128
    t = np.arange(search_end) / sample_rate
    corrected = segment[:search_end] * np.exp(-2j * np.pi * cfo_hz * t)
    corr = np.abs(np.correlate(corrected, LTS_TIME, mode='valid')) ** 2
    energy = np.convolve(np.abs(corrected) ** 2, np.ones(64), mode='valid')
    ref_energy = np.vdot(LTS_TIME, LTS_TIME).real
    quality = corr / np.maximum(energy * ref_energy, 1e-20)
    pair_quality = np.minimum(quality[lo:hi + 1], quality[lo + 64:hi + 65])
    best = int(np.argmax(pair_quality))
    if pair_quality[best] < 0.15:
        return None
    return lo + best

# --- SCHMIDL & COX ---
def schmidl_cox_metric(iq_signal, N=_SC_N, W=_SC_W):
    L = len(iq_signal)
    
    # 1. Productos cruzados y energía (directamente sobre la señal cruda)
    prod = np.conj(iq_signal[:-N]) * iq_signal[N:]
    energy_before = np.abs(iq_signal[:-N]) ** 2
    energy_after = np.abs(iq_signal[N:]) ** 2
    
    # 2. Integración
    ventana = np.ones(W)
    P = np.convolve(prod, ventana, mode='valid')
    R_before = np.convolve(energy_before, ventana, mode='valid')
    R = np.convolve(energy_after, ventana, mode='valid')
    
    # 3. Recorte
    P = P[:L - 2 * N]
    R_before = R_before[:L - 2 * N]
    R = R[:L - 2 * N]
    
    # 4. Métrica final
    M = np.abs(P) ** 2 / np.maximum(R_before * R, 1e-20)
    
    return M, P, R

# Decodificador Viterbi rate 1/2
# Polinomios generadores: g0=133, g1=171 (octal) = 0b1011011, 0b1111001
# Constraint length K=7, memoria=6

def viterbi_decode(bits, K=7, g0=0b1011011, g1=0b1111001, soft=False):
    """
    Decodificador Viterbi para codigo convolucional rate 1/2.
    Entrada: pares [b0, b1, ...]. En modo soft, valores reales con
    signo positivo para 1 y negativo para 0; la magnitud expresa confianza.
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
                if soft:
                    # Término de la distancia euclídea que depende de la
                    # hipótesis; los términos constantes se cancelan.
                    dist = -(2 * b0 - 1) * rx0 - (2 * b1 - 1) * rx1
                else:
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
    bits = np.asarray(bits)
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
    result = np.empty(NCBPS, dtype=bits.dtype)
    for k in range(NCBPS):
        i = (NCBPS // 16) * (k % 16) + k // 16
        result[k] = bits_step1[i]
    
    return result


def pilot_polarities(n_symbols):
    """Polaridad de pilotos 802.11a/g, desde SIGNAL (símbolo 0)."""
    reg = [1] * 7
    sequence = np.empty(n_symbols, dtype=np.int8)
    for n in range(n_symbols):
        bit = reg[3] ^ reg[6]
        sequence[n] = 1 - 2 * bit
        reg = [bit] + reg[:-1]
    return sequence


def channel_equalizer(H, active_bins):
    """Devuelve ganancias finitas aun si alguna subportadora está en un nulo."""
    channel = H[active_bins]
    power = np.abs(channel) ** 2
    if not np.all(np.isfinite(power)):
        return None, None
    typical_power = np.median(power)
    if typical_power <= 1e-12:
        return None, None
    power_floor = typical_power * 1e-6
    weights = np.zeros_like(H)
    weights[active_bins] = np.conj(channel) / (power + power_floor)
    return weights, power_floor


def calculate_evm(data_rx, pilots_rx, pilots_ref, modulation, data_bins,
                  pilot_bins, valid_data, valid_pilots):
    """EVM de símbolos OFDM completos, referida a potencia ideal unitaria.

    Es una estimación por decisiones duras, no una medición de conformidad:
    una decisión errónea puede subestimar el error real.
    """
    levels_by_mod = {
        'BPSK': np.array([-1, 1]),
        'QPSK': np.array([-1, 1]),
        '16-QAM': np.array([-3, -1, 1, 3]),
        '64-QAM': np.array([-7, -5, -3, -1, 1, 3, 5, 7]),
    }
    if modulation not in levels_by_mod or not np.any(valid_data):
        return None
    base = levels_by_mod[modulation]
    power = np.mean(base ** 2) * (1 if modulation == 'BPSK' else 2)
    levels = base / np.sqrt(power)
    received = data_rx[:, valid_data]
    ideal_i = levels[np.argmin(np.abs(received.real[..., None] - levels), axis=-1)]
    if modulation == 'BPSK':
        ideal_q = 0
    else:
        ideal_q = levels[np.argmin(np.abs(received.imag[..., None] - levels), axis=-1)]
    ideal = ideal_i + 1j * ideal_q

    # Las constelaciones 802.11 tienen potencia nominal 1 (los pilotos ±1).
    # Escalar con la potencia recibida reduciría artificialmente el EVM con ruido.
    data_error = np.abs(received - ideal)
    pilot_error = np.abs(pilots_rx[:, valid_pilots] - pilots_ref[:, valid_pilots])
    all_error = np.concatenate((data_error, pilot_error), axis=1)
    if not np.all(np.isfinite(all_error)):
        return None

    to_db = lambda error: 20 * np.log10(np.maximum(error, 1e-10))
    subc_rms = np.sqrt(np.mean(all_error ** 2, axis=0))
    subc_peak = np.max(all_error, axis=0)
    bins = np.r_[np.asarray(data_bins)[valid_data], np.asarray(pilot_bins)[valid_pilots]]
    signed_bins = np.where(bins < 32, bins, bins - 64)
    order = np.argsort(signed_bins)
    return {
        'subc_x': signed_bins[order],
        'subc_rms': to_db(subc_rms[order]),
        'subc_peak': to_db(subc_peak[order]),
        'sym_rms': to_db(np.sqrt(np.mean(all_error ** 2, axis=1))),
        'sym_peak': to_db(np.max(all_error, axis=1)),
    }


def is_ht_mixed_signal(data_symbols, valid_data):
    """Reconoce los dos HT-SIG en Q-BPSK que siguen al L-SIG de 6 Mb/s."""
    if len(data_symbols) < 2 or np.count_nonzero(valid_data) < 24:
        return False
    points = data_symbols[:2, valid_data]
    q_dominant = np.mean(np.abs(points.imag) > np.abs(points.real), axis=1)
    q_power = np.mean(points.imag ** 2, axis=1)
    return bool(np.all(q_dominant >= 0.75) and np.all(q_power > 0.2))

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
        self._generation = 0
        self._pending_blocks = deque(maxlen=32)
        self._completed_results = deque(maxlen=8)
        self._tail_iq = np.empty(0, dtype=np.complex64)
        self.dropped_blocks = 0
        self._next_display_at = 0.0
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
        # Descartamos cualquier resultado anterior para no mostrar datos de otra
        # configuración/sesión al arrancar.
        with self._lock:
            self._generation += 1
            self._pending_blocks.clear()
            self._completed_results.clear()
            self._tail_iq = np.empty(0, dtype=np.complex64)
            self.dropped_blocks = 0
            self._next_display_at = 0.0
            self.nuevos_datos_listos = False
            self.last_heavy_results = {}
            self.ultimo_puntos_corr = None
            self.ultimo_wifi_metrics = {}
            self.ultimo_evm_data = None
            self.ultimo_S_data = None
            self.ultimo_chunk_norm = None
            self.ultimo_M_norm = None

    def procesar(self, muestras_iq):
        with self._lock:
            if muestras_iq is None:
                self._generation += 1
                self._pending_blocks.clear()
                self._completed_results.clear()
                self._tail_iq = np.empty(0, dtype=np.complex64)
                self.nuevos_datos_listos = False
                self._next_display_at = 0.0
                return None

            if len(muestras_iq):
                current = np.asarray(muestras_iq).copy()
                previous = self._tail_iq
                tail_size = int(self.sample_rate * 0.004)
                self._tail_iq = np.concatenate((previous, current))[-tail_size:].copy()
                if len(self._pending_blocks) == self._pending_blocks.maxlen:
                    self.dropped_blocks += 1
                self._pending_blocks.append((previous, current, self._generation))
                if not self.is_processing:
                    self._start_next_locked()

            if self._completed_results:
                result = self._completed_results.popleft()
                self.nuevos_datos_listos = bool(self._completed_results)
                return result
            return None

    def _start_next_locked(self):
        if not self._pending_blocks:
            return
        previous, current, generation = self._pending_blocks.popleft()
        self.is_processing = True
        threading.Thread(
            target=self._process_queued_block,
            args=(previous, current, generation),
            daemon=True,
        ).start()

    def _process_queued_block(self, previous, current, generation):
        try:
            block = np.concatenate((previous, current))
            self._procesar_fondo(block, len(previous), generation)
        except Exception:
            logging.exception("Error al procesar un bloque WiFi")
        finally:
            with self._lock:
                self.is_processing = False
                self._start_next_locked()

    def _procesar_fondo(self, bloque_iq: np.ndarray, min_new_end=0, generation=None):
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
            max_energia = np.max(energia_suave) if len(energia_suave) else 0
            
            chunk_trigger = None
            envolvente_preambulo = None  # |preámbulo| para visualizar estructura STS/LTS en Q3
            wifi_metrics = {}
            

            energia_norm = energia_suave / max_energia if max_energia > 0 else np.zeros_like(energia_suave)
            en_burst_raw = energia_norm > 0.3
            
            # --- PROTECCIÓN CONTRA FALSO FIN DE BURST ---
            # Aplicamos cierre morfológico: Si hay una caída de energía menor a 100 muestras (5us)
            # producida por fading o ruido, se "rellena" conectando el burst.
            margen_cierre = 101
            mascara_extendida = np.pad(en_burst_raw, margen_cierre)
            en_burst = minimum_filter1d(
                maximum_filter1d(mascara_extendida, size=margen_cierre),
                size=margen_cierre,
            )[margen_cierre:-margen_cierre]
            
            cambios = np.diff(np.pad(en_burst.astype(np.int8), 1))
            inicios_burst = np.flatnonzero(cambios == 1)
            fines_burst = np.flatnonzero(cambios == -1)
            # El burst abierto al final se volverá a ver completo en el bloque
            # siguiente, gracias al solapamiento de muestras.
            bursts = [
                (ini, fin) for ini, fin in zip(inicios_burst, fines_burst)
                if fin > min_new_end and (fin < len(bloque_iq) or not en_burst[-1])
            ]

            # ---  RECORTE DEL BURST (chunk_norm) ---
            margen_muestras = int(10e-6 * self.sample_rate) # 10 us de margen (200 muestras a 20MHz)
            chunk_norm = energia_norm # Por defecto (si no hay bursts) mandamos todo

            inicio_recorte = 0
            
            if bursts:
                inicio_recorte = max(0, bursts[0][0] - margen_muestras)
                fin_recorte = min(len(energia_norm), bursts[0][1] + margen_muestras)
                chunk_norm = energia_norm[inicio_recorte:fin_recorte]

            if bursts:
                for ini, fin in bursts:
                    inicio_recorte = max(0, ini - margen_muestras)
                    fin_recorte = min(len(energia_norm), fin + margen_muestras)
                    chunk_norm = energia_norm[inicio_recorte:fin_recorte]
                    ini_ext = max(0, ini - int(4e-6 * self.sample_rate))
                    segmento = bloque_iq[ini_ext:fin]
                    
                    if len(segmento) < (_SC_N + _SC_W):
                        continue
                        
                    # 2. BÚSQUEDA FINA - Schmidl & Cox
                    M, P, R = schmidl_cox_metric(segmento)
                    if len(M) == 0:
                        continue
                    max_energy = np.max(R)
                    if max_energy <= 0:
                        continue
                    # La forma simétrica está acotada por 1; en silencio un
                    # pico de correlación aleatorio no debe fijar la escala.
                    M_norm = np.where(R > 0.1 * max_energy, M, 0)

                    # Una meseta STS debe ser continua; picos aislados de ruido
                    # no constituyen un preámbulo. La posición fina sale del LTS.
                    indices_sts = np.flatnonzero(M_norm > 0.7)
                    runs = np.split(indices_sts, np.flatnonzero(np.diff(indices_sts) > 1) + 1)
                    runs = [run for run in runs if len(run) >= 32]
                    if runs:
                        plateau = max(runs, key=len)
                        cfo_rad = np.angle(np.mean(P[plateau])) / _SC_N
                        cfo_hz = cfo_rad * self.sample_rate / (2 * np.pi)
                        lts_local = find_lts_start(
                            segmento, cfo_hz, self.sample_rate,
                            int(plateau[0]), int(plateau[-1]),
                        )
                        if lts_local is None or lts_local < 192:
                            continue
                        muestra_abs = ini_ext + lts_local - 192
                        
                        margen_visual = 150
                        inicio_visual = max(0, muestra_abs - margen_visual)
                        
                        if fin - muestra_abs >= 400:
                            chunk_trigger = bloque_iq[inicio_visual : inicio_visual + fs].copy()
                            # El umbral de energía puede caer dentro del último
                            # símbolo OFDM; conservar un margen para completarlo.
                            frame_end = min(len(bloque_iq), fin + 80)
                            frame = bloque_iq[muestra_abs : frame_end]
                            wifi_metrics['cfo'] = cfo_hz

                            # El SIGNAL termina en la muestra 400 del paquete.
                            if len(frame) < 400:
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

                            if not np.isfinite(P_senal) or P_senal <= 0 or not np.isfinite(P_ruido):
                                continue
                            snr_lineal = P_senal / max(P_ruido, P_senal * 1e-12)
                            snr_db     = 10 * np.log10(snr_lineal)
                            wifi_metrics['snr'] = snr_db

                            P_frame = np.mean(np.abs(frame_corr) ** 2)
                            if not np.isfinite(P_frame) or P_frame <= 0:
                                continue
                            gain_agc = 1.0 / np.sqrt(P_frame)
                            frame_norm = frame_corr * gain_agc

                            # Extraemos 400 muestras del preámbulo (STS + GI2 + LTS + SIGNAL)
                            # para visualizar su estructura en el cuadrante Q3.
                            if len(frame_norm) >= 400:
                                envolvente_preambulo = np.abs(frame_norm[:400])
                            
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
                            equalizer, channel_power_floor = channel_equalizer(H, activas_lts)
                            if equalizer is None:
                                continue

                            # Demodulacion del campo SIGNAL
                            # Ubicacion: justo despues de STS + GI2 + 2*LTS
                            N_CP_SIGNAL = 16   # prefijo ciclico
                            N_FFT       = 64

                            inicio_signal = N_STS + N_GI2 + 2 * N_LTS
                            signal_sym = frame_norm[inicio_signal + N_CP_SIGNAL : inicio_signal + N_CP_SIGNAL + N_FFT]

                            # FFT y ecualizacion
                            S = np.fft.fft(signal_sym, N_FFT)
                            S_eq = np.zeros(N_FFT, dtype=complex)
                            S_eq[activas_lts] = S[activas_lts] * equalizer[activas_lts]

                            # SIGNAL también lleva pilotos (polaridad p_0).
                            # Compensar su fase común antes de decidir RATE/LENGTH.
                            pilot_idx_ordered = [7, 21, 43, 57]
                            pilot_ref = np.array([1, -1, 1, 1])
                            valid_pilots = np.abs(H[pilot_idx_ordered]) ** 2 > channel_power_floor
                            if not np.any(valid_pilots):
                                continue
                            signal_rotation = S_eq[pilot_idx_ordered] * np.conj(pilot_ref)
                            signal_phase = np.angle(np.mean(signal_rotation[valid_pilots]))
                            S_eq[activas_lts] *= np.exp(-1j * signal_phase)

                            data_idx  = list(range(38, 64)) + list(range(1, 27))
                            pilot_idx = [43, 57, 7, 21]
                            data_idx  = [i for i in data_idx if i not in pilot_idx]

                            S_data = S_eq[data_idx]

                            # El L-SIG usa BPSK. El signo indica el bit y la
                            # magnitud, ponderada por |H|², su confiabilidad.
                            signal_power = np.abs(H[data_idx]) ** 2
                            typical_power = np.median(np.abs(H[activas_lts]) ** 2)
                            reliability = np.clip(signal_power / typical_power, 0, 4)
                            soft_symbols = deinterleave_signal(S_data.real * reliability)
                            bits_decoded = viterbi_decode(soft_symbols, soft=True)

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
                            reservado_ok = (info_bits[4] == 0)

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
                                'reservado_ok': reservado_ok,
                                'tail_ok': tail_ok
                            })

                            # --- VALIDACIÓN ESTRICTA L-SIG (Rechazo de Falsos Positivos) ---
                            # RATE, paridad, bit reservado y tail deben ser válidos.
                            if mod == "?" or not paridad_ok or not reservado_ok or not tail_ok:
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
                            
                            # Un burst truncado no debe publicarse como una
                            # medición EVM de la trama completa.
                            if N_simbolos_max < N_simbolos_exacto:
                                continue
                            N_simbolos = N_simbolos_exacto

                            # Correccion de fase simbolo a simbolo usando pilotos
                            # Pilotos en indices FFT: 7, 21, 43, 57 (+7, +21, -21, -7)
                            # Valores de referencia: [+1, +1, +1, -1] * secuencia_pn

                            pn = pilot_polarities(N_simbolos + 1)
                            # Según IEEE 802.11, P_{-21, -7, 7, 21} = {1, 1, 1, -1}
                            # En orden de FFT (+7, +21, -21, -7) esto es [1, -1, 1, 1]
                            pilot_ref = np.array([1, -1, 1, 1])  # subportadoras +7,+21,-21,-7

                            pilot_idx_ordered = [7, 21, 43, 57]  # orden en FFT
                            valid_pilots = np.abs(H[pilot_idx_ordered]) ** 2 > channel_power_floor
                            if not np.any(valid_pilots):
                                continue
                            valid_data = np.abs(H[data_idx]) ** 2 > channel_power_floor

                            constelacion_corr = []
                            pilots_corr = []
                            pilots_ideales = []
                            ht_detected = False
                            for k in range(N_simbolos):
                                offset = inicio_datos + k * (N_CP + N_FFT)
                                simbolo = frame_norm[offset + N_CP : offset + N_CP + N_FFT]
                                S = np.fft.fft(simbolo, N_FFT)
                                S_eq = S.copy()

                                # Ecualizar datos
                                S_eq[data_idx] = S[data_idx] * equalizer[data_idx]

                                # Estimar fase residual con los pilotos
                                pn_k = pn[k + 1]  # SIGNAL ocupa el símbolo 0
                                pilots_rx = S[pilot_idx_ordered] * equalizer[pilot_idx_ordered]
                                pilots_exp = pilot_ref * pn_k
                                rot = pilots_rx * np.conj(pilots_exp)
                                fase_residual = np.angle(np.mean(rot[valid_pilots]))

                                # Corregir fase en las subportadoras de datos y pilotos
                                S_eq[data_idx] *= np.exp(-1j * fase_residual)
                                pilots_rx_corr = pilots_rx * np.exp(-1j * fase_residual)

                                constelacion_corr.append(S_eq[data_idx])
                                pilots_corr.append(pilots_rx_corr)
                                pilots_ideales.append(pilots_exp)

                                if mbps == 6 and k == 1 and is_ht_mixed_signal(
                                    np.asarray(constelacion_corr), valid_data,
                                ):
                                    ht_detected = True
                                    break

                            if ht_detected:
                                # El L-SIG HT imita 6 Mb/s; ignorar esta trama
                                # y seguir buscando otra a/g dentro del bloque.
                                wifi_metrics = {}
                                puntos_corr = None
                                S_data = None
                                M_norm = np.array([])
                                chunk_norm = energia_norm
                                chunk_trigger = None
                                inicio_recorte = 0
                                continue

                            constelacion_corr = np.array(constelacion_corr)
                            puntos_corr = constelacion_corr.flatten()
                            pilots_corr = np.array(pilots_corr)
                            pilots_ideales = np.array(pilots_ideales)

                            # --- CÁLCULO DE EVM ---
                            evm_data = calculate_evm(
                                constelacion_corr, pilots_corr, pilots_ideales, mod,
                                data_idx, pilot_idx_ordered, valid_data, valid_pilots,
                            )
                            if evm_data is not None:
                                with self._lock:
                                    if generation is None or generation == self._generation:
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
            potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk_psd, n=fs)))**2 / fs
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            
            # Interpolamos el bin DC para tapar el spike de hardware
            centro = fs // 2
            PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0
            
            with self._lock:
                if generation is not None and generation != self._generation:
                    return
                # Mantener la última constelación válida hasta encontrar otra.
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
                    'metricas': {
                        'inicio_recorte': inicio_recorte,
                        'wifi_metrics': self.ultimo_wifi_metrics,
                        'dropped_iq_blocks': self.dropped_blocks,
                    },
                    'evm_data': self.ultimo_evm_data,
                }
                self.last_heavy_results = resultados
                now = time.monotonic()
                if now >= self._next_display_at:
                    self._completed_results.append(resultados)
                    self.nuevos_datos_listos = True
                    self._next_display_at = now + 0.05
            
        except Exception:
            logging.exception("Error en el análisis de una trama WiFi")
