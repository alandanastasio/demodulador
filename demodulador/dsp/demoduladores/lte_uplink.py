import numpy as np
import threading
import time
from .base import DemoduladorBase
from scipy import signal
from .lte_downlink import generar_pss_time, generar_sss
from numba import jit

@jit(nopython=True)
def generar_dmrs_lte_jit(u, v, alpha_idx, M_sc):
    N_zc = M_sc
    while N_zc > 3:
        is_prime = True
        for i in range(2, int(N_zc**0.5) + 1):
            if N_zc % i == 0:
                is_prime = False
                break
        if is_prime: break
        N_zc -= 1

    q_bar = N_zc * (u + 1) / 31.0
    q0 = int(np.floor(q_bar + 0.5))
    
    # Numba doesn't support ** for float base and negative exponent easily in some versions,
    # but (-1)**int is safe. Let's do it manually just in case.
    if int(np.floor(2 * q_bar)) % 2 == 0:
        sign = 1
    else:
        sign = -1
        
    q1 = q0 + sign
    
    q_cands = np.zeros(2, dtype=np.int32)
    q_cands[0] = q0
    num_cands = 1
    if q1 != q0 and q1 > 0:
        q_cands[1] = q1
        num_cands = 2
        
    q = q_cands[v % num_cands]

    k_vec = np.arange(M_sc)
    n_zc = k_vec % N_zc
    r_bar = np.exp(-1j * np.pi * q * n_zc * (n_zc + 1) / N_zc)
    alpha = 2 * np.pi * alpha_idx / 12
    return r_bar * np.exp(1j * alpha * k_vec)

@jit(nopython=True)
def fit_phase_slope_jit(phases):
    M = len(phases)
    sum_x = (M - 1) * M / 2.0
    sum_x2 = (M - 1) * M * (2 * M - 1) / 6.0
    sum_y = np.sum(phases)
    sum_xy = 0.0
    for i in range(M):
        sum_xy += i * phases[i]
    
    denominator = M * sum_x2 - sum_x**2
    if denominator == 0:
        return 0.0
    return (M * sum_xy - sum_x * sum_y) / denominator

@jit(nopython=True)
def fit_phase_residual_jit(phases):
    M = len(phases)
    sum_x = (M - 1) * M / 2.0
    sum_x2 = (M - 1) * M * (2 * M - 1) / 6.0
    sum_y = np.sum(phases)
    sum_xy = 0.0
    for i in range(M):
        sum_xy += i * phases[i]
    
    denominator = M * sum_x2 - sum_x**2
    if denominator == 0:
        return 0.0
    slope = (M * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / M
    
    residual = 0.0
    for i in range(M):
        fit = slope * i + intercept
        residual += (phases[i] - fit)**2
    return np.sqrt(residual / M)

@jit(nopython=True)
def resolver_ambiguedad_qpsk_jit(s_time_c):
    rots = np.array([1, 1j, -1, -1j], dtype=np.complex128)
    mejor_rot = 1.0 + 0j
    max_qpsk = -1
    for i in range(len(rots)):
        r = rots[i]
        test = s_time_c * r
        score = 0
        for j in range(len(test)):
            if np.abs(test[j].real) > 0.5 and np.abs(test[j].imag) > 0.5:
                score += 1
        if score > max_qpsk:
            max_qpsk = score
            mejor_rot = r
    return mejor_rot

class DemoduladorLTEUplink(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 30.72e6
        self.fft_size = 2048

        self.Tu = 2048
        self.cp_len_1 = 160   # CP del símbolo 0 (más largo)
        self.cp_len_2 = 144   # CP de los símbolos 1-6

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
        
        self._ultimo_dmrs_bueno = np.array([])
        self._ultimo_pusch_bueno = np.array([])

        self.occupied_subcarriers = np.array([])
        self.rb_count = 100
        self.half_shift = None
        self.CP_pattern = None

        # --- Máquina de Estados ---
        self.estado = 'WAITING_DL'
        self.cell_id_guardada = None
        self.n_id_1 = None
        self.n_id_2 = None

    @property
    def id(self):
        return "lte_uplink"

    @property
    def nombre_mostrar(self):
        return "LTE Uplink (SC-FDMA)"

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
        
        self.half_shift = np.exp(-1j * np.pi * np.arange(self.Tu) / self.Tu)
        self.CP_pattern = [self.cp_len_1] + [self.cp_len_2]*6 + [self.cp_len_1] + [self.cp_len_2]*6
        
        self.estado = 'WAITING_DL'
        self.cell_id_guardada = None

    def _generar_dmrs_lte(self, u, v, alpha_idx, M_sc):
        N_zc = M_sc
        while N_zc > 3:
            if all(N_zc % i for i in range(2, int(N_zc**0.5) + 1)): break
            N_zc -= 1

        q_bar = N_zc * (u + 1) / 31.0
        q0 = int(np.floor(q_bar + 0.5))
        sign = (-1) ** int(np.floor(2 * q_bar))
        q1 = q0 + sign
        
        # Secuencias base permitidas para este grupo
        q_cands = [q0]
        if q1 != q0 and q1 > 0:
            q_cands.append(q1)
            
        q = q_cands[v % len(q_cands)]

        k_vec = np.arange(M_sc)
        n_zc = k_vec % N_zc
        r_bar = np.exp(-1j * np.pi * q * n_zc * (n_zc + 1) / N_zc)
        alpha = 2 * np.pi * alpha_idx / 12
        return r_bar * np.exp(1j * alpha * k_vec)

    # ------------------------------------------------------------------
    #  PROCESAMIENTO PESADO (corre en un hilo separado)
    # ------------------------------------------------------------------
    def _procesar_heavy_thread(self, chunk):
        try:
            Tu = self.Tu
            cp1 = self.cp_len_1
            cp2 = self.cp_len_2
            slot_len = cp1 + Tu + 6 * (cp2 + Tu)

            # ---- PSD para UI ----
            ui_fs = self.fft_size
            chunk_psd = chunk[:ui_fs].copy() if len(chunk) >= ui_fs else np.pad(chunk, (0, ui_fs - len(chunk)))
            chunk_psd -= np.mean(chunk_psd)
            potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk_psd, n=ui_fs))) ** 2 / ui_fs
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            centro_psd = ui_fs // 2
            PSD[centro_psd] = (PSD[centro_psd - 1] + PSD[centro_psd + 1]) / 2.0
            rf_chunk_ui = chunk[:ui_fs] if len(chunk) >= ui_fs else np.pad(chunk, (0, ui_fs - len(chunk)))

            # ---- Variables por defecto para resultados ----
            dmrs_plot = self._ultimo_dmrs_bueno
            pusch_plot = self._ultimo_pusch_bueno
            cfo_hz = 0
            cfo_fraccional_hz = 0
            evm_dmrs_str = "--"
            evm_pusch_str = "--"
            pwr_dmrs_str = "--"
            pwr_pusch_str = "--"

            # =========================================================
            # ESTADO 1: Escuchar Downlink
            # =========================================================
            if self.estado == 'WAITING_DL':
                pss_time, _ = generar_pss_time(self.fft_size)
                max_val = 0
                mejor_N_id_2 = -1
                mejor_pico = -1
                l_limit = min(len(chunk), int(self.sample_rate * 0.02))
                
                if l_limit > self.fft_size:
                    for i in range(3):
                        corr = signal.correlate(chunk[:l_limit], pss_time[i], mode="valid", method="fft")
                        corr_abs = np.abs(corr)
                        pico_local = np.argmax(corr_abs)
                        val = corr_abs[pico_local]
                        if val > max_val:
                            max_val = val
                            mejor_N_id_2 = i
                            mejor_pico = pico_local

                    mean_corr = np.mean(np.abs(corr))
                    umbral_pss = 4.0 * mean_corr
                    
                    if max_val > umbral_pss:
                        inicio_sss = mejor_pico - Tu - cp2
                        if inicio_sss >= 0:
                            sss_f = np.fft.fftshift(np.fft.fft(chunk[inicio_sss: inicio_sss + Tu]))
                            centro = self.fft_size // 2
                            idx_sss = list(range(centro - 31, centro)) + list(range(centro + 1, centro + 32))
                            sss_rx = sss_f[idx_sss]

                            mejor_corr_sss = 0
                            mejor_N_id_1 = -1
                            for subf in (0, 5):
                                for n_id_1 in range(168):
                                    d_ref = generar_sss(n_id_1, mejor_N_id_2, subf)
                                    c = np.abs(np.vdot(d_ref, sss_rx))
                                    if c > mejor_corr_sss:
                                        mejor_corr_sss = c
                                        mejor_N_id_1 = n_id_1

                            if mejor_N_id_1 != -1:
                                self.cell_id_guardada = 3 * mejor_N_id_1 + mejor_N_id_2
                                self.n_id_1 = mejor_N_id_1
                                self.n_id_2 = mejor_N_id_2
                                self.estado = 'WAITING_UL'
                                print(f"[UPLINK SNIFFER] DL Cell ID = {self.cell_id_guardada}. Cambie a freq UL.")

            # =========================================================
            # ESTADO 2: Escuchar Uplink - Demodular PUSCH completo
            # =========================================================
            elif self.estado in ('WAITING_UL', 'DMRS_CHECKPOINT'):
                if self.cell_id_guardada is None:
                    # Fallback si por algun motivo forzamos el estado sin Cell ID
                    self.cell_id_guardada = 5 
                
                # 1. S&C y CFO
                # Aplicamos shift de +7.5kHz para CFO estimation
                t_arr = np.arange(len(chunk))
                chunk_shifted = chunk * np.exp(1j * 2 * np.pi * 7500 * t_arr / self.sample_rate)
                
                prod = chunk_shifted[:-Tu] * np.conjugate(chunk_shifted[Tu:])
                # Usar cp_largo para mas robustez como vimos en notebook (usamos cp1=20 en vez de cp2=18 en el convolver)
                cp_corr = signal.fftconvolve(prod, np.ones(cp1), mode='valid')
                
                num_slots = len(cp_corr) // slot_len
                if num_slots >= 1:
                    folded = np.zeros(slot_len, dtype=complex)
                    for s in range(num_slots):
                        folded += cp_corr[s * slot_len: (s + 1) * slot_len]

                    template = np.zeros(slot_len)
                    sym_starts = [0]
                    offset = cp1 + Tu
                    for _ in range(6):
                        sym_starts.append(offset)
                        offset += cp2 + Tu
                    for off in sym_starts:
                        template[off] = 1.0

                    sync = np.fft.ifft(np.fft.fft(np.abs(folded)) * np.conj(np.fft.fft(template)))
                    a0_basto = int(np.argmax(np.abs(sync)))
                    
                    # CFO: S&C fase
                    fase_cfo = np.angle(folded[a0_basto])
                    cfo_estimado = -fase_cfo * self.sample_rate / (2 * np.pi * Tu)
                    
                    # Como el S&C se hizo sobre chunk_shifted (+7.5kHz), cfo_estimado ya absorbió el offset.
                    # Corregimos solo el error de hardware para que half_shift (-7.5kHz) en la FFT deje todo en DC.
                    # NOTA: multiplicamos por -1j para compensar el error positivo.
                    cfo_fraccional_hz = cfo_estimado
                    cfo_hz = cfo_estimado
                    chunk_corregido = chunk * np.exp(-1j * 2 * np.pi * cfo_estimado * t_arr / self.sample_rate)

                    # 2. Timing Fino
                    mejor_a0 = a0_basto
                    mejor_metric = float('inf')
                    for d in range(-15, 16):
                        a_test = a0_basto + d
                        if a_test < 0 or a_test + Tu > len(chunk_corregido): continue
                        sym = chunk_corregido[a_test : a_test+Tu]
                        sf = np.fft.fftshift(np.fft.fft(sym * self.half_shift))
                        s_pow = np.abs(sf)**2
                        # Metric: power outside occupied subcarriers
                        out_pow = np.sum(s_pow) - np.sum(s_pow[self.occupied_subcarriers])
                        if out_pow < mejor_metric:
                            mejor_metric = out_pow
                            mejor_a0 = a_test

                    # 3. Macro/Micro Sync y Extraer 14 símbolos
                    # Primero extraemos con CP asumiendo que arranca en simbolo 0
                    syms_rx = []
                    for i in range(14):
                        delta = (cp1 - cp2) * (i // 7)
                        start = mejor_a0 + i * (Tu + cp2) + delta
                        if start >= 0 and start + Tu <= len(chunk_corregido):
                            s_t = chunk_corregido[start:start+Tu]
                            s_f = np.fft.fftshift(np.fft.fft(s_t * self.half_shift))
                            syms_rx.append(s_f[self.occupied_subcarriers])
                            
                    if len(syms_rx) == 14:
                        syms_rx = np.array(syms_rx)
                        M_sc = self.rb_count * 12
                        if M_sc >= 12:
                            
                            # 4. Encontrar DMRS (por Coef. de Variacion minimo)
                            cv = np.std(np.abs(syms_rx), axis=1) / (np.mean(np.abs(syms_rx), axis=1) + 1e-9)
                            # Buscamos los 2 minimos que tengan separacion de al menos 4 simbolos
                            idx_sort = np.argsort(cv)
                            p1, p2 = 3, 10 # Default fallback
                            for i in range(len(idx_sort)):
                                for j in range(i+1, len(idx_sort)):
                                    if abs(idx_sort[i] - idx_sort[j]) >= 4:
                                        p1, p2 = min(idx_sort[i], idx_sort[j]), max(idx_sort[i], idx_sort[j])
                                        break
                                else:
                                    continue
                                break

                            # Deducir simbolo_arranque (p1 deberia ser el DMRS 1 que esta en index 3 de la subtrama)
                            simbolo_arranque = (3 - p1) % 14
                            CP_alineado = self.CP_pattern[simbolo_arranque:] + self.CP_pattern[:simbolo_arranque]

                            # ZC Search
                            u = self.cell_id_guardada % 30
                            
                            # Buscar v minimizando el residuo (con alpha=0)
                            mejor_v = 0
                            min_res = float('inf')
                            for v in (0, 1):
                                ref = generar_dmrs_lte_jit(u, v, 0, M_sc)
                                H_est = syms_rx[p1] * np.conjugate(ref)
                                res = fit_phase_residual_jit(np.unwrap(np.angle(H_est)))
                                if res < min_res:
                                    min_res = res
                                    mejor_v = v
                                    
                            mejor_H1, mejor_alpha1, mejor_slope1 = None, 0, 0
                            mejor_v1 = mejor_v
                            min_slope1 = float('inf')
                            for alpha in range(12):
                                ref = generar_dmrs_lte_jit(u, mejor_v1, alpha, M_sc)
                                H_est = syms_rx[p1] * np.conjugate(ref)
                                slope = np.abs(fit_phase_slope_jit(np.unwrap(np.angle(H_est))))
                                if slope < min_slope1:
                                    min_slope1 = slope
                                    mejor_H1 = H_est
                                    mejor_alpha1 = alpha

                            mejor_H2, mejor_alpha2, mejor_slope2 = None, 0, 0
                            mejor_v2 = mejor_v
                            min_slope2 = float('inf')
                            for alpha in range(12):
                                ref = generar_dmrs_lte_jit(u, mejor_v2, alpha, M_sc)
                                H_est = syms_rx[p2] * np.conjugate(ref)
                                slope = np.abs(fit_phase_slope_jit(np.unwrap(np.angle(H_est))))
                                if slope < min_slope2:
                                    min_slope2 = slope
                                    mejor_H2 = H_est
                                    mejor_alpha2 = alpha

                            # STO (Micro sync)
                            slope_rad_sc = fit_phase_slope_jit(np.unwrap(np.angle(mejor_H1)))
                            error_muestras = -slope_rad_sc * Tu / (2 * np.pi)
                            
                            # Traducir el error de timing al inicio de la trama (a0_perfecto)
                            start_p1_loop1 = mejor_a0 + p1 * (Tu + cp2) + (cp1 - cp2) * (p1 // 7)
                            start_p1_true = start_p1_loop1 + error_muestras
                            delta_cp_p1_loop2 = sum(CP_alineado[1:p1+1]) if p1 > 0 else 0
                            a0_perfecto = int(np.round(start_p1_true - p1 * Tu - delta_cp_p1_loop2))
                            if a0_perfecto < 0:
                                print(f"⚠️ a0_perfecto = {a0_perfecto} (negativo). La señal arranca demasiado cerca del inicio del buffer.")
                                a0_perfecto = 0

                            # 8. Re-extraer
                            syms_rx_p = []
                            for i in range(14):
                                delta_cp = sum(CP_alineado[1:i+1]) if i > 0 else 0
                                start = a0_perfecto + i * Tu + delta_cp
                                if start >= 0 and start + Tu <= len(chunk_corregido):
                                    s_t = chunk_corregido[start:start+Tu]
                                    s_f = np.fft.fftshift(np.fft.fft(s_t * self.half_shift))
                                    syms_rx_p.append(s_f[self.occupied_subcarriers])
                            
                            if len(syms_rx_p) == 14:
                                syms_rx_p = np.array(syms_rx_p)
                                ref1 = generar_dmrs_lte_jit(u, mejor_v1, mejor_alpha1, M_sc)
                                ref2 = generar_dmrs_lte_jit(u, mejor_v2, mejor_alpha2, M_sc)
                                H1_p = syms_rx_p[p1] * np.conjugate(ref1)
                                H2_p = syms_rx_p[p2] * np.conjugate(ref2)
                                
                                # 9. Hibrido
                                x = np.arange(M_sc)
                                p_mag_1_p = np.polyfit(x, np.abs(H1_p), 3)
                                p_mag_2_p = np.polyfit(x, np.abs(H2_p), 3)
                                slope_1_p = fit_phase_slope_jit(np.unwrap(np.angle(H1_p)))
                                slope_2_p = fit_phase_slope_jit(np.unwrap(np.angle(H2_p)))
                                
                                mag_1_smooth_p = np.polyval(p_mag_1_p, x)
                                mag_2_smooth_p = np.polyval(p_mag_2_p, x)

                                const_pts = []
                                dmrs_pts = []
                                
                                # Process symbols
                                for i in range(14):
                                    # Interpolate channel
                                    t = np.clip((i - p1) / (p2 - p1), -0.5, 1.5)
                                    mag = mag_1_smooth_p * (1 - t) + mag_2_smooth_p * t
                                    slope = slope_1_p * (1 - t) + slope_2_p * t
                                    H_i = mag * np.exp(1j * slope * (x - M_sc / 2.0))
                                    
                                    # Equalize
                                    s_eq = syms_rx_p[i] / (H_i + 1e-9)
                                    
                                    if i == p1 or i == p2:
                                        # El usuario desea ver la "rueda" (secuencia ZC ecualizada) en lugar del clúster derotado.
                                        # Simplemente normalizamos la potencia del símbolo ecualizado (s_eq) y lo graficamos.
                                        dmrs_norm = s_eq / (np.sqrt(np.mean(np.abs(s_eq)**2)) + 1e-9)
                                        dmrs_pts.extend(dmrs_norm)
                                    else:
                                        # SC-FDMA IDFT
                                        s_time = np.fft.ifft(s_eq) * np.sqrt(M_sc)
                                        
                                        # Normalize power for Viterbi-Viterbi and QPSK ambiguity resolution
                                        rms = np.sqrt(np.mean(np.abs(s_time)**2)) + 1e-9
                                        s_time_n = s_time * (np.sqrt(2.0) / rms)
                                        
                                        # VV
                                        ph_eq = np.angle(np.mean(s_time_n**4)) / 4
                                        s_time_c = s_time_n * np.exp(-1j * (ph_eq - np.pi/4))
                                        
                                        # Ambigüedad 90 (Fuerza bruta heuristica acelerada)
                                        mejor_rot = resolver_ambiguedad_qpsk_jit(s_time_c)
                                        const_pts.extend(s_time_c * mejor_rot)
                                        
                                if len(const_pts) > 0:
                                    const_pts = np.array(const_pts)
                                    dmrs_pts = np.array(dmrs_pts)
                                    
                                    self._ultimo_pusch_bueno = const_pts
                                    self._ultimo_dmrs_bueno = dmrs_pts
                                    
                                    pusch_plot = const_pts
                                    dmrs_plot = dmrs_pts
                                    
                                    # Calc EVM
                                    # PUSCH (Normalizado a 1)
                                    pusch_norm = pusch_plot / (np.sqrt(np.mean(np.abs(pusch_plot)**2)) + 1e-9)
                                    ideal_pusch = np.sign(pusch_norm.real) + 1j*np.sign(pusch_norm.imag)
                                    ideal_pusch /= np.sqrt(2)
                                    evm_pusch = np.sqrt(np.mean(np.abs(pusch_norm - ideal_pusch)**2)) * 100
                                    evm_pusch_str = f"{evm_pusch:.1f}%"
                                    
                                    # DMRS (Normalizado a 1)
                                    dmrs_norm = dmrs_plot / (np.mean(np.abs(dmrs_plot)) + 1e-9)
                                    evm_dmrs = np.sqrt(np.mean(np.abs(dmrs_norm - 1.0)**2)) * 100
                                    evm_dmrs_str = f"{evm_dmrs:.1f}%"
                                    
                                    pwr_pusch_str = f"{10*np.log10(np.mean(np.abs(pusch_plot)**2) + 1e-9):.1f} dB"
                                    pwr_dmrs_str = f"{10*np.log10(np.mean(np.abs(dmrs_plot)**2) + 1e-9):.1f} dB"
                                    
                                    self.estado = 'DMRS_CHECKPOINT'

            # ---- Empaquetar resultados ----
            self.ultimo_lte_metrics = {
                'pss_found': self.cell_id_guardada is not None,
                'cfo_hz': cfo_hz,
                'cfo_fraccional_hz': cfo_fraccional_hz,
                'cell_id': self.cell_id_guardada,
                'estado': self.estado,
                'frame_summary': {
                    'DMRS': (evm_dmrs_str, pwr_dmrs_str, f"{self.rb_count} RB"),
                    'PUSCH': (evm_pusch_str, pwr_pusch_str, f"{self.rb_count} RB"),
                }
            }

            resultados = {
                'psd_rf': PSD,
                'rf_chunk': rf_chunk_ui,
                'mpx_time': np.array([]),
                'audio_time_L': pusch_plot.real if len(pusch_plot) > 0 else np.array([]),
                'audio_time_R': pusch_plot.imag if len(pusch_plot) > 0 else np.array([]),
                'psd_mpx': np.array([]),
                'f_axis_mpx': np.array([]),
                'metricas': {
                    'lte_metrics': self.ultimo_lte_metrics,
                    'pss_pts': dmrs_plot,
                },
                'evm_data': None,
            }

            with self._lock:
                self.last_heavy_results = resultados
                self.nuevos_datos_listos = True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[UPLINK] Error: {e}")
        finally:
            self.is_processing = False
            self.proxima_captura = time.time() + self.pausa_entre_snapshots

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        if muestras_iq is None or len(muestras_iq) == 0:
            return None

        if self.is_processing:
            if self.nuevos_datos_listos:
                with self._lock:
                    self.nuevos_datos_listos = False
                    return self.last_heavy_results
            return None

        ahora = time.time()
        if ahora < self.proxima_captura:
            return None

        self.buffer_medicion.append(muestras_iq)
        self.muestras_acumuladas += len(muestras_iq)

        muestras_necesarias = int(self.sample_rate * 0.02)
        if self.muestras_acumuladas >= muestras_necesarias:
            chunk = np.concatenate(self.buffer_medicion)
            self.buffer_medicion = []
            self.muestras_acumuladas = 0

            self.is_processing = True
            threading.Thread(
                target=self._procesar_heavy_thread,
                args=(chunk,),
                daemon=True,
            ).start()

        if self.nuevos_datos_listos:
            with self._lock:
                self.nuevos_datos_listos = False
                return self.last_heavy_results

        return None
