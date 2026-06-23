import numpy as np
import threading
from scipy.ndimage import uniform_filter1d
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
                next_s, b0, b1 = conv_output(inp, state)
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
    s = max(NBPSC // 2, 1)
    
    # Permutacion inversa de la segunda permutacion
    # j -> i: invertir j = s*floor(i/s) + (i + floor(16*i/NCBPS)) % s
    j = np.arange(NCBPS)
    i_step1 = np.zeros(NCBPS, dtype=int)
    for i in range(NCBPS):
        jj = (s * (i // s) + (i + int(16 * i / NCBPS)) % s) % NCBPS
        i_step1[jj] = i
    bits_step1 = bits[i_step1]
    
    # Permutacion inversa de la primera permutacion
    # k -> i: invertir k = (NCBPS/16)*(i%16) + floor(i/16)
    k = np.arange(NCBPS)
    i_step2 = np.zeros(NCBPS, dtype=int)
    for i in range(NCBPS):
        kk = (NCBPS // 16) * (i % 16) + i // 16
        i_step2[kk] = i
    bits_step2 = bits_step1[i_step2]
    
    return bits_step2

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
        
        time.sleep(0.5)
        muestras_necesarias = int(self.sample_rate * 0.001) #capturamos 1.000us
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
            puntos_corr = None

            # --- 0. LIMPIEZA DE HARDWARE ---
            # Eliminamos la fuga del oscilador local (DC Offset) de todo el bloque
            bloque_iq = bloque_iq - np.mean(bloque_iq)

            # 1. BÚSQUEDA GRUESA (Energía)
            energia = np.abs(bloque_iq) ** 2
            energia_suave = uniform_filter1d(energia, size=50)
            max_energia = np.max(energia_suave)
            
            chunk_trigger = None
            envolvente_preambulo = None  # |preámbulo| para visualizar estructura STS/LTS en Q3
            

            energia_norm = energia_suave / max_energia
            en_burst = energia_norm > 0.3
            cambios = np.diff(en_burst.astype(int))
            inicios_burst = np.where(cambios == 1)[0]
            fines_burst   = np.where(cambios == -1)[0]

            # Si la senal empieza ya dentro de un burst, agregar inicio en 0
            if en_burst[0]:
                inicios_burst = np.concatenate(([0], inicios_burst))

            n_bursts = min(len(inicios_burst), len(fines_burst))
            inicios_burst = inicios_burst[:n_bursts]
            fines_burst   = fines_burst[:n_bursts]
            print(f"Se encontraron {n_bursts} bursts")

            # ---  RECORTE DEL BURST (chunk_norm) ---
            margen_muestras = int(10e-6 * self.sample_rate) # 10 us de margen (200 muestras a 20MHz)
            chunk_norm = energia_norm # Por defecto (si no hay bursts) mandamos todo
            
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
                    
                    if len(segmento) < _SC_W:
                        continue
                        
                    # 2. BÚSQUEDA FINA - Schmidl & Cox
                    M, P, R = schmidl_cox_metric(segmento)
                    if len(M) == 0:
                        continue
                    M_norm = M / np.max(M)

                    # Buscamos el primer índice donde la correlación normalizada supera 0.7
                    indices_sts = np.where(M_norm > 0.7)[0]
                    
                    if len(indices_sts) > 0:
                        muestra_local = indices_sts[0]
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

                            # CFO en Hz
                            cfo_hz = cfo_rad * self.sample_rate / (2 * np.pi)

                            print(f"CFO estimado: {cfo_hz:+.1f} Hz  ({cfo_rad*1e3:+.3f} mrad/muestra)")

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

                            print(f"SNR estimada: {snr_db:.1f} dB")

                            P_frame = np.mean(np.abs(frame_corr) ** 2)
                            gain_agc = 1.0 / np.sqrt(P_frame)
                            frame_norm = frame_corr * gain_agc

                            # Extraemos 400 muestras del preámbulo (STS + GI2 + LTS + SIGNAL)
                            # para visualizar su estructura en el cuadrante Q3.
                            if len(frame_norm) >= 400:
                                envolvente_preambulo = np.abs(frame_norm[:400])
                            
                            # LTS correcto segun estandar 802.11-2007, tabla 18-7
                            # Orden natural: subportadora 0, +1, ..., +31, -32, ..., -1
                            LTS_FREQ = np.array([
                                0, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1,
                                -1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,
                                -1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 0
                            ], dtype=complex)
                            print(f"LTS_REF tiene {len(LTS_FREQ)} elementos")

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
                            print(f"CFO fino: {cfo_fino_hz:+.1f} Hz")

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
                            bits_raw = (S_data.real < 0).astype(int)

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

                            # Demodulacion de los simbolos de datos (64-QAM)
                            N_CP  = 16
                            N_FFT = 64

                            # Subportadoras de datos (48) y pilotos (4)
                            data_idx  = list(range(38, 64)) + list(range(1, 27))
                            pilot_idx = [43, 57, 7, 21]
                            data_idx  = [i for i in data_idx if i not in pilot_idx]

                            # Inicio de los simbolos de datos: STS + GI2 + 2*LTS + SIGNAL
                            inicio_datos = N_STS + N_GI2 + 2 * N_LTS + (N_CP + N_FFT)

                            # Cuantos simbolos entran en el frame
                            muestras_disponibles = len(frame_norm) - inicio_datos
                            N_simbolos = muestras_disponibles // (N_CP + N_FFT)
                            print(f"Simbolos de datos disponibles: {N_simbolos}")

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
                            pilot_ref = np.array([1, 1, 1, -1])  # subportadoras +7,+21,-21,-7

                            pilot_idx_ordered = [7, 21, 43, 57]  # orden en FFT

                            constelacion_corr = []
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

                                # Corregir fase en las subportadoras de datos
                                S_eq[data_idx] *= np.exp(-1j * fase_residual)

                                constelacion_corr.append(S_eq[data_idx])

                            constelacion_corr = np.array(constelacion_corr)
                            puntos_corr = constelacion_corr.flatten()

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
                'rf_chunk': chunk_norm,
                'mpx_time': M_norm,  
                'audio_time_L': puntos_corr.real if puntos_corr is not None else None,
                'audio_time_R': puntos_corr.imag if puntos_corr is not None else None,
                'psd_mpx': S_data.real if 'S_data' in locals() else None,
                'f_axis_mpx': S_data.imag if 'S_data' in locals() else None,
            }

            with self._lock:
                self.last_heavy_results = resultados
                self.nuevos_datos_listos = True
            
        finally:
            self.is_processing = False