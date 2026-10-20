from fractions import Fraction
from collections import deque
from threading import Lock, Thread

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, firwin
from .base import DemoduladorBase
from ..streaming_resampler import StreamingResampler
from numba import jit


@jit(nopython=True, nogil=True)
def correr_pll_38k(piloto, fs, fase_inicial, integral_inicial, avance_fase):
    N = len(piloto)
    portadora_38k = np.zeros(N)
    
    f_center = 19000.0
    w_center = 2.0 * np.pi * f_center / fs
    fase = fase_inicial
    
    alpha = 0.05
    beta = 0.001
    filtro_integral = integral_inicial
    
    for i in range(N):
        # 1. Detector de fase
        error = piloto[i] * -np.sin(fase)
        
        # 2. Filtro de lazo PI
        filtro_integral += beta * error
        ajuste_fase = alpha * error + filtro_integral
        
        # El piloto filtrado llega retrasado respecto de L-R. Compensar esa
        # diferencia antes de duplicar la fase; el signo fija L y R según
        # la fase de la subportadora estéreo respecto del piloto.
        portadora_38k[i] = -np.sin(2.0 * (fase + avance_fase))
        
        # 4. Actualizar NCO
        fase += w_center + ajuste_fase
        if fase > 2.0 * np.pi:
            fase -= 2.0 * np.pi
        elif fase < 0.0:
            fase += 2.0 * np.pi
            
    return portadora_38k, fase, filtro_integral

class DemoduladorWBFMAudio(DemoduladorBase):
    def __init__(self):
        # --- Buffers y memoria (Reemplaza las globales de tu state) ---
        self.fm_buffer = np.empty(0, dtype=np.complex64)
        self.fm_last_iq = 1+0j
        
        # Filtros FIR (caché)
        self.bb_lpf_kernel = None
        self.mpx_lpf_kernel = None
        
        # Condiciones iniciales de filtros (Z-States)
        self.lpf_zi = None
        self.mpx_zi = None
        self.deemph_zi = None
        
        # Métricas suavizadas
        self.avg_pico_max = None
        self.avg_pico_min = None
        self.avg_pico_rms = None
        self.avg_dc_offset = None
        
        # Parámetros operativos
        self.sample_rate = 10e6
        self.fft_size = 4096

        # Filtros y Estados para FM Estéreo ---
        self.stereo_lpr_kernel = None # LPF para L+R (15 kHz)
        self.stereo_pilot_kernel = None # BPF para Piloto (19 kHz)
        self.stereo_lmr_kernel = None # BPF para L-R modulado (23-53 kHz)
        self.stereo_audio_lpf_kernel = None # LPF para L-R demodulado (15 kHz)
        
        self.lpr_zi = None
        self.pilot_zi = None
        self.lmr_zi = None
        self.audio_lpf_zi = None
        self.lpr_delay_samples = 0
        self.lpr_delay_buffer = np.empty(0)
        
        # Variables de estado para el PLL de 19kHz -> 38kHz
        self.pll_fase = 0.0
        self.pll_filtro_integral = 0.0
        self.stereo_blend = 0.0
        self.audio_gain = None
        self.audio_highpass_zi = None
        self._work_lock = Lock()
        self._pending_blocks = deque()
        self._ready_results = deque(maxlen=2)
        self._worker_active = False
        self._generation = 0
        self._reset_after_gap = False
        self.result_callback = None
        self.dropped_iq_blocks = 0

    @property
    def id(self): return "wbfm_audio"

    @property
    def nombre_mostrar(self): return "FM: escucha"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        
        hw_sr = int(self.sample_rate)
        factor_bajada = int(hw_sr / 300000)
        nueva_fs = hw_sr / factor_bajada 
        audio_ratio = Fraction(48_000 * factor_bajada, hw_sr)
        self.factor_bajada = factor_bajada
        self.audio_ratio = audio_ratio
        self.rf_resampler = StreamingResampler(1, factor_bajada)
        self.audio_resampler = StreamingResampler(audio_ratio.numerator, audio_ratio.denominator)
        self.deemph_b, self.deemph_a = butter(1, 3183 / (48000/2), btype='low')
        self.highpass_b, self.highpass_a = butter(1, 20, fs=48_000,
                                                btype='highpass')
        
        # Calculamos los filtros FIR una sola vez
        self.bb_lpf_kernel = firwin(201, 200e3 / (hw_sr / 2.0))
        self.mpx_lpf_kernel = firwin(65, 80000, fs=nueva_fs)
        
        # Limpiamos los estados de memoria
        self.lpf_zi = None
        self.mpx_zi = None
        self.deemph_l_zi = None
        self.deemph_r_zi = None
        self.fm_buffer = np.empty(0, dtype=np.complex64)
        self.fm_last_iq = 1 + 0j
        self.avg_pico_max = self.avg_pico_min = None
        self.avg_pico_rms = self.avg_dc_offset = None
        self.stereo_blend = 0.0
        self.audio_gain = None
        self.audio_highpass_zi = None

        # ---  Diseñar Filtros Estéreo ---
        # freq_nyquist = nueva_fs / 2.0 (usaremos fs = nueva_fs que es ~300k o 240k)
        freq_nyq = nueva_fs / 2.0
        
        # 1. Filtro L+R (Pasa Bajos 15 kHz) - 101 taps
        self.stereo_lpr_kernel = firwin(101, 15000 / freq_nyq)
        
        # 2. Filtro Tono Piloto (Pasa Banda 18.5k - 19.5k) - 201 taps (muy selectivo)
        self.stereo_pilot_kernel = firwin(201, [18500 / freq_nyq, 19500 / freq_nyq], pass_zero=False)
        
        # 3. Filtro L-R Modulado (Pasa Banda 23k - 53k) - 101 taps
        self.stereo_lmr_kernel = firwin(101, [23000 / freq_nyq, 53000 / freq_nyq], pass_zero=False)
        
        # 4. Filtro para limpiar el L-R ya bajado a banda base (Pasa Bajos 15 kHz)
        self.stereo_audio_lpf_kernel = firwin(101, 15000 / freq_nyq)

        # L-R atraviesa un FIR más después de la mezcla. Igualar el retardo
        # de L+R evita que las dos ramas se sumen con distinta fase de audio.
        self.lpr_delay_samples = (
            len(self.stereo_lmr_kernel) + len(self.stereo_audio_lpf_kernel)
            - len(self.stereo_lpr_kernel) - 1
        ) // 2
        self.lpr_delay_buffer = np.zeros(self.lpr_delay_samples)
        self.pilot_phase_advance = (2 * np.pi * 19000 / nueva_fs *
                                    (len(self.stereo_pilot_kernel)
                                     - len(self.stereo_lmr_kernel)) / 2)
        
        # Limpiar estados
        self.lpr_zi = None
        self.pilot_zi = None
        self.lmr_zi = None
        self.audio_lpf_zi = None
        self.pll_fase = 0.0
        self.pll_filtro_integral = 0.0
        # Compilar el PLL fuera del callback de recepción: la primera llamada
        # de Numba puede costar mucho más que los siguientes bloques de 100 ms.
        correr_pll_38k(np.zeros(8), nueva_fs, 0.0, 0.0, 0.0)

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        if muestras_iq is None:
            with self._work_lock:
                self._generation += 1
                self._pending_blocks.clear()
                self._ready_results.clear()
                self._reset_after_gap = True
            self.fm_buffer = np.empty(0, dtype=np.complex64)
            return None
        if len(muestras_iq):
            self.fm_buffer = np.concatenate((self.fm_buffer, muestras_iq))
        hw_sr = int(self.sample_rate)
        chunk_size_hw = int(hw_sr * 0.1)

        with self._work_lock:
            while len(self.fm_buffer) >= chunk_size_hw:
                chunk_hw = self.fm_buffer[:chunk_size_hw].copy()
                self.fm_buffer = self.fm_buffer[chunk_size_hw:]
                if len(self._pending_blocks) >= 2:
                    self.dropped_iq_blocks += len(self._pending_blocks) + int(self._worker_active)
                    self._pending_blocks.clear()
                    self._ready_results.clear()
                    self._generation += 1
                    self._reset_after_gap = True
                self._pending_blocks.append((self._generation, chunk_hw))
            if self._pending_blocks and not self._worker_active:
                self._worker_active = True
                Thread(target=self._worker_loop, daemon=True,
                       name='wbfm-audio-dsp').start()
            if self.result_callback is None and self._ready_results:
                return self._ready_results.popleft()
            return None

    def _worker_loop(self):
        while True:
            with self._work_lock:
                if not self._pending_blocks:
                    self._worker_active = False
                    return
                generation, chunk_hw = self._pending_blocks.popleft()
                reset = self._reset_after_gap
                self._reset_after_gap = False
            if reset:
                self._reset_signal_path()
            try:
                result = self._procesar_bloque(chunk_hw)
            except Exception:
                import traceback
                traceback.print_exc()
                with self._work_lock:
                    self._worker_active = False
                    self._pending_blocks.clear()
                    self._reset_after_gap = True
                return
            with self._work_lock:
                if generation == self._generation:
                    if 'metricas' in result:
                        result['metricas']['dropped_iq_blocks'] = self.dropped_iq_blocks
                    callback = self.result_callback
                    if callback is None:
                        self._ready_results.append(result)
                else:
                    callback = None
            if callback is not None:
                try:
                    callback(result)
                except Exception:
                    import traceback
                    traceback.print_exc()

    def _reset_signal_path(self):
        self.lpf_zi = self.mpx_zi = None
        self.lpr_zi = self.pilot_zi = self.lmr_zi = self.audio_lpf_zi = None
        self.deemph_l_zi = self.deemph_r_zi = None
        self.rf_resampler = StreamingResampler(1, self.factor_bajada)
        self.audio_resampler = StreamingResampler(self.audio_ratio.numerator,
                                                  self.audio_ratio.denominator)
        self.fm_last_iq = 1 + 0j
        self.lpr_delay_buffer.fill(0)
        self.pll_fase = self.pll_filtro_integral = 0.0
        self.stereo_blend = 0.0
        self.audio_gain = None
        self.audio_highpass_zi = None
        self.avg_pico_max = self.avg_pico_min = None
        self.avg_pico_rms = self.avg_dc_offset = None

    def _procesar_bloque(self, chunk_hw: np.ndarray) -> dict:
        hw_sr = int(self.sample_rate)
        
        factor_bajada = int(hw_sr / 300000)
        nueva_fs = hw_sr / factor_bajada 
        
        # === PASO 1 y 2: BANDA BASE Y FILTRADO FIR ===
        bb = chunk_hw - np.mean(chunk_hw)
        if self.lpf_zi is None:
            self.lpf_zi = lfilter_zi(self.bb_lpf_kernel, [1.0]) * bb[0]
        bb_filt, self.lpf_zi = lfilter(self.bb_lpf_kernel, [1.0], bb, zi=self.lpf_zi)
        
        # === PASO 3: DOWNSAMPLING ===
        bb_resample = self.rf_resampler.process(bb_filt)
        
        # --- ESPECTRO RF CRUDO ---
        fs_fft = self.fft_size
        rf_fft_chunk = None
        PSD = None
        if len(bb_resample) >= fs_fft:
            rf_fft_chunk = bb_resample[:fs_fft] - np.mean(bb_resample[:fs_fft])
            potencia = np.abs(np.fft.fftshift(np.fft.fft(rf_fft_chunk)))**2 / fs_fft
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            centro = fs_fft // 2
            PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0
            
        # === PASO 4: DEMODULACIÓN FM (Ángulo del producto conjugado) ===
        chunk_with_last = np.insert(bb_resample, 0, self.fm_last_iq)
        self.fm_last_iq = bb_resample[-1]
        msj = np.angle(chunk_with_last[1:] * np.conjugate(chunk_with_last[:-1])) * (nueva_fs / (2*np.pi))
        demod_khz_raw = msj / 1000.0
        
        # === FILTRO MPX ===
        if self.mpx_zi is None:
            self.mpx_zi = lfilter_zi(self.mpx_lpf_kernel, [1.0]) * demod_khz_raw[0]
        demod_khz_clean, self.mpx_zi = lfilter(self.mpx_lpf_kernel, [1.0], demod_khz_raw, zi=self.mpx_zi)
        
        # === PASO 5: EXTRACCIÓN ESTÉREO ===
        
        # A. Extraer L+R (Mono)
        if self.lpr_zi is None:
            self.lpr_zi = lfilter_zi(self.stereo_lpr_kernel, [1.0]) * demod_khz_clean[0]
        senal_L_plus_R, self.lpr_zi = lfilter(self.stereo_lpr_kernel, [1.0], demod_khz_clean, zi=self.lpr_zi)
        if self.lpr_delay_samples:
            delayed_lpr = np.concatenate((self.lpr_delay_buffer, senal_L_plus_R))
            self.lpr_delay_buffer = delayed_lpr[-self.lpr_delay_samples:].copy()
            senal_L_plus_R = delayed_lpr[:len(senal_L_plus_R)]
        
        # B. Extraer Piloto 19 kHz
        if self.pilot_zi is None:
            self.pilot_zi = lfilter_zi(self.stereo_pilot_kernel, [1.0]) * demod_khz_clean[0]
        piloto_19k, self.pilot_zi = lfilter(self.stereo_pilot_kernel, [1.0], demod_khz_clean, zi=self.pilot_zi)

        # El piloto debe sobresalir del ruido vecino antes de abrir L-R.
        # Sin él, el PLL gira sin referencia y agrega ruido estéreo al mono.
        probe = demod_khz_clean[:min(len(demod_khz_clean), 30000)]
        pilot_spectrum = np.abs(np.fft.rfft(probe * np.hanning(len(probe))))**2
        pilot_freqs = np.fft.rfftfreq(len(probe), 1 / nueva_fs)
        pilot_peak = np.max(pilot_spectrum[np.abs(pilot_freqs - 19000) <= 100])
        adjacent = pilot_spectrum[(pilot_freqs >= 17000) & (pilot_freqs <= 21000)
                                  & (np.abs(pilot_freqs - 19000) >= 300)]
        pilot_snr_db = 10 * np.log10((pilot_peak + 1e-20)
                                    / (np.median(adjacent) + 1e-20))
        pilot_rms = np.sqrt(np.mean(piloto_19k**2))
        mpx_rms = np.sqrt(np.mean(demod_khz_clean**2))
        stereo_locked = pilot_snr_db >= 12 and pilot_rms >= 0.005 * mpx_rms
        target_blend = 1.0 if stereo_locked else 0.0
        coefficient = 0.9 if stereo_locked else 0.5
        self.stereo_blend += coefficient * (target_blend - self.stereo_blend)
        
        # C. Extraer Banda L-R Modulada (23-53 kHz)
        if self.lmr_zi is None:
            self.lmr_zi = lfilter_zi(self.stereo_lmr_kernel, [1.0]) * demod_khz_clean[0]
        lmr_modulado, self.lmr_zi = lfilter(self.stereo_lmr_kernel, [1.0], demod_khz_clean, zi=self.lmr_zi)
        
        # D. Regenerar Portadora 38 kHz con PLL
        portadora_38k, self.pll_fase, self.pll_filtro_integral = correr_pll_38k(
            piloto_19k, nueva_fs, self.pll_fase, self.pll_filtro_integral,
            self.pilot_phase_advance,
        )
        
        # E. Demodular L-R (Multiplicador Síncrono)
        # Se multiplica por 2.0 para recuperar la amplitud perdida en el DSB
        lmr_mezclado = lmr_modulado * portadora_38k * 2.0
        
        # F. Limpiar el L-R demodulado (Pasa Bajos 15 kHz)
        if self.audio_lpf_zi is None:
            self.audio_lpf_zi = lfilter_zi(self.stereo_audio_lpf_kernel, [1.0]) * lmr_mezclado[0]
        senal_L_minus_R, self.audio_lpf_zi = lfilter(self.stereo_audio_lpf_kernel, [1.0], lmr_mezclado, zi=self.audio_lpf_zi)
        senal_L_minus_R *= self.stereo_blend
        
        # G. MATRIZ ESTÉREO
        canal_L = (senal_L_plus_R + senal_L_minus_R) / 2.0
        canal_R = (senal_L_plus_R - senal_L_minus_R) / 2.0
        
        # --- ESPECTRO MPX (Se usa demod_khz_clean original para ver todo el panorama) ---
        fs_fft = self.fft_size
        PSD_audio, f_axis_audio = None, None
        if len(demod_khz_clean) >= fs_fft:
            potencia_audio = np.abs(np.fft.fft(demod_khz_clean[:fs_fft]))**2 / fs_fft
            mitad = fs_fft // 2
            PSD_audio = 10.0 * np.log10(np.maximum(potencia_audio[:mitad], 1e-12))
            f_axis_audio = np.fft.fftfreq(fs_fft, d=1 / nueva_fs)[:mitad] / 1e3
            
        # --- MÉTRICAS DE DESVIACIÓN ---
        inst_pico_max = np.percentile(demod_khz_clean, 99)
        inst_pico_min = np.percentile(demod_khz_clean, 1)
        desv_rms_true = np.sqrt(np.mean(demod_khz_clean**2))
        inst_pico_rms = max(abs(inst_pico_max), abs(inst_pico_min)) / np.sqrt(2)
        inst_dc_offset = np.mean(demod_khz_clean)
        
        if self.avg_pico_max is None:
            self.avg_pico_max = inst_pico_max
            self.avg_pico_min = inst_pico_min
            self.avg_pico_rms = inst_pico_rms
            self.avg_dc_offset = inst_dc_offset
        else:
            alpha = 0.3
            self.avg_pico_max = alpha * inst_pico_max + (1 - alpha) * self.avg_pico_max
            self.avg_pico_min = alpha * inst_pico_min + (1 - alpha) * self.avg_pico_min
            self.avg_pico_rms = alpha * inst_pico_rms + (1 - alpha) * self.avg_pico_rms
            self.avg_dc_offset = alpha * inst_dc_offset + (1 - alpha) * self.avg_dc_offset
            
        fm_metrics = {
            'pico_max': self.avg_pico_max,
            'pico_min': self.avg_pico_min,
            'rms': desv_rms_true,
            'pico_rms': self.avg_pico_rms,
            'dc_offset': self.avg_dc_offset
        }

       # --- DE-ÉNFASIS PARA CANALES L Y R ---
        stereo_48k = self.audio_resampler.process(np.column_stack((canal_L, canal_R)))
        L_48k, R_48k = stereo_48k.T
        b_aud, a_aud = self.deemph_b, self.deemph_a
        
        if self.deemph_l_zi is None:
            self.deemph_l_zi = lfilter_zi(b_aud, a_aud) * L_48k[0]
            self.deemph_r_zi = lfilter_zi(b_aud, a_aud) * R_48k[0]
            
        L_filtrado, self.deemph_l_zi = lfilter(b_aud, a_aud, L_48k, zi=self.deemph_l_zi)
        R_filtrado, self.deemph_r_zi = lfilter(b_aud, a_aud, R_48k, zi=self.deemph_r_zi)

        audio_filtered = np.column_stack((L_filtrado, R_filtrado))
        if self.audio_highpass_zi is None:
            self.audio_highpass_zi = (lfilter_zi(self.highpass_b, self.highpass_a)[:, None]
                                       * audio_filtered[0])
        audio_filtered, self.audio_highpass_zi = lfilter(
            self.highpass_b, self.highpass_a, audio_filtered, axis=0,
            zi=self.audio_highpass_zi)
        L_filtrado, R_filtrado = audio_filtered.T

        # Ganancia común y suave para ambos canales: evitar saltos de volumen
        # de un bloque al siguiente y conservar el balance estéreo.
        max_val = max(np.max(np.abs(L_filtrado)), np.max(np.abs(R_filtrado)), 15.0)
        target_gain = 1.0 / max_val
        old_gain = target_gain if self.audio_gain is None else self.audio_gain
        new_gain = min(target_gain, old_gain + 0.05 * (target_gain - old_gain))
        self.audio_gain = new_gain
        gain_ramp = np.linspace(old_gain, new_gain, len(L_filtrado))
        audio_stereo = np.column_stack((L_filtrado, R_filtrado))
        audio_stereo = np.clip(audio_stereo * gain_ramp[:, None], -1, 1).astype(np.float32)
        fm_metrics.update({
            'rms_L': float(np.sqrt(np.mean(audio_stereo[:, 0]**2))),
            'rms_R': float(np.sqrt(np.mean(audio_stereo[:, 1]**2))),
            'pilot_snr_db': float(pilot_snr_db),
            'stereo_locked': bool(stereo_locked),
            'stereo_blend': float(self.stereo_blend),
        })
        
        # --- PREPARAR SALIDA PARA GRÁFICOS (Canales estéreo) ---
        muestras_10ms = int(nueva_fs * 0.01)
        # Usamos canal_L y canal_R puros (sin de-énfasis) para ver el nivel de baseband
        t_axis_audio = np.linspace(0, 10, muestras_10ms)
        
        # Empaquetamos todo
        return {
            'psd_rf': PSD,
            'rf_chunk': rf_fft_chunk,
            'psd_mpx': PSD_audio,
            'f_axis_mpx': f_axis_audio,
            'audio_time_L': canal_L[:muestras_10ms], 
            'audio_time_R': canal_R[:muestras_10ms], 
            't_axis_audio': t_axis_audio,
            'mpx_time': demod_khz_clean[:muestras_10ms],
            'audio_out': audio_stereo,
            'metricas': fm_metrics
        }
