import numpy as np
from scipy.signal import butter, lfilter, resample_poly, lfilter_zi, firwin
from .base import DemoduladorBase
from numba import jit


@jit(nopython=True)
def correr_pll_38k(piloto, fs, fase_inicial, integral_inicial):
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
        
        # 3. Generar 38 kHz (doble de la fase)
        # Usamos seno porque el filtro de 19kHz desplaza la fase 90 grados
        portadora_38k[i] = np.sin(2.0 * fase)
        
        # 4. Actualizar NCO
        fase += w_center + ajuste_fase
        if fase > 2.0 * np.pi:
            fase -= 2.0 * np.pi
        elif fase < 0.0:
            fase += 2.0 * np.pi
            
    return portadora_38k, fase, filtro_integral

class DemoduladorWBFM(DemoduladorBase):
    def __init__(self):
        # --- Buffers y memoria (Reemplaza las globales de tu state) ---
        self.fm_buffer = np.array([], dtype=np.complex128)
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
        
        # Variables de estado para el PLL de 19kHz -> 38kHz
        self.pll_fase = 0.0
        self.pll_filtro_integral = 0.0

    @property
    def id(self): return "wbfm"

    @property
    def nombre_mostrar(self): return "WBFM (Radio Comercial)"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        
        hw_sr = int(self.sample_rate)
        factor_bajada = int(hw_sr / 300000)
        nueva_fs = hw_sr / factor_bajada 
        
        # Calculamos los filtros FIR una sola vez
        self.bb_lpf_kernel = firwin(201, 200e3 / (hw_sr / 2.0))
        self.mpx_lpf_kernel = firwin(65, 80000, fs=nueva_fs)
        
        # Limpiamos los estados de memoria
        self.lpf_zi = None
        self.mpx_zi = None
        self.deemph_l_zi = None
        self.deemph_r_zi = None
        self.fm_buffer = np.array([], dtype=np.complex128)

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
        
        # Limpiar estados
        self.lpr_zi = None
        self.pilot_zi = None
        self.lmr_zi = None
        self.audio_lpf_zi = None
        self.pll_fase = 0.0
        self.pll_filtro_integral = 0.0

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        self.fm_buffer = np.append(self.fm_buffer, muestras_iq)
        hw_sr = int(self.sample_rate)
        
        # Bloques de 100 ms para baja latencia
        chunk_size_hw = int(hw_sr * 0.1) 
        
        if len(self.fm_buffer) < chunk_size_hw:
            # Si no juntamos 100ms todavía, no devolvemos nada
            return None 
            
        chunk_hw = self.fm_buffer[:chunk_size_hw]
        self.fm_buffer = self.fm_buffer[chunk_size_hw:]
        
        factor_bajada = int(hw_sr / 300000)
        nueva_fs = hw_sr / factor_bajada 
        
        # === PASO 1 y 2: BANDA BASE Y FILTRADO FIR ===
        bb = chunk_hw - np.mean(chunk_hw)
        if self.lpf_zi is None:
            self.lpf_zi = lfilter_zi(self.bb_lpf_kernel, [1.0]) * bb[0]
        bb_filt, self.lpf_zi = lfilter(self.bb_lpf_kernel, [1.0], bb, zi=self.lpf_zi)
        
        # === PASO 3: DOWNSAMPLING ===
        bb_resample = resample_poly(bb_filt, up=1, down=factor_bajada, window=('kaiser', 8.6))
        
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
        
        # B. Extraer Piloto 19 kHz
        if self.pilot_zi is None:
            self.pilot_zi = lfilter_zi(self.stereo_pilot_kernel, [1.0]) * demod_khz_clean[0]
        piloto_19k, self.pilot_zi = lfilter(self.stereo_pilot_kernel, [1.0], demod_khz_clean, zi=self.pilot_zi)
        
        # C. Extraer Banda L-R Modulada (23-53 kHz)
        if self.lmr_zi is None:
            self.lmr_zi = lfilter_zi(self.stereo_lmr_kernel, [1.0]) * demod_khz_clean[0]
        lmr_modulado, self.lmr_zi = lfilter(self.stereo_lmr_kernel, [1.0], demod_khz_clean, zi=self.lmr_zi)
        
        # D. Regenerar Portadora 38 kHz con PLL
        portadora_38k, self.pll_fase, self.pll_filtro_integral = correr_pll_38k(
            piloto_19k, nueva_fs, self.pll_fase, self.pll_filtro_integral
        )
        
        # E. Demodular L-R (Multiplicador Síncrono)
        # Se multiplica por 2.0 para recuperar la amplitud perdida en el DSB
        lmr_mezclado = lmr_modulado * portadora_38k * 2.0
        
        # F. Limpiar el L-R demodulado (Pasa Bajos 15 kHz)
        if self.audio_lpf_zi is None:
            self.audio_lpf_zi = lfilter_zi(self.stereo_audio_lpf_kernel, [1.0]) * lmr_mezclado[0]
        senal_L_minus_R, self.audio_lpf_zi = lfilter(self.stereo_audio_lpf_kernel, [1.0], lmr_mezclado, zi=self.audio_lpf_zi)
        
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
            f_axis_audio = np.linspace(0, (nueva_fs/2)/1e3, mitad)
            
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
        L_48k = resample_poly(canal_L, 4, 25)
        R_48k = resample_poly(canal_R, 4, 25)
        
        # Filtro de de-énfasis de 50µs (Constante de tiempo para Argentina/Europa)
        b_aud, a_aud = butter(1, 3183 / (48000/2), btype='low')
        
        if self.deemph_l_zi is None:
            self.deemph_l_zi = lfilter_zi(b_aud, a_aud) * L_48k[0]
            self.deemph_r_zi = lfilter_zi(b_aud, a_aud) * R_48k[0]
            
        L_filtrado, self.deemph_l_zi = lfilter(b_aud, a_aud, L_48k, zi=self.deemph_l_zi)
        R_filtrado, self.deemph_r_zi = lfilter(b_aud, a_aud, R_48k, zi=self.deemph_r_zi)
        
        L_filtrado = L_filtrado - np.mean(L_filtrado)
        R_filtrado = R_filtrado - np.mean(R_filtrado)
        
        # Buscamos el pico máximo entre los DOS canales para normalizar balanceado
        max_val = max(np.max(np.abs(L_filtrado)), np.max(np.abs(R_filtrado)), 15.0)
        
        audio_L_norm = np.float32(L_filtrado / max_val) if max_val > 0 else np.float32(L_filtrado)
        audio_R_norm = np.float32(R_filtrado / max_val) if max_val > 0 else np.float32(R_filtrado)
        
        # Juntamos L y R en un array de 2D (Columnas)
        audio_stereo = np.column_stack((audio_L_norm, audio_R_norm))
        
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
            'audio_out': audio_stereo, # Sigue saliendo solo el L por los parlantes por ahora
            'metricas': fm_metrics
        }