import numpy as np
from scipy.signal import resample_poly, firwin
from .base import DemoduladorBase
from numba import jit

# --- PLL COMPILADO EN C (Velocidad extrema) ---
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
        error = piloto[i] * -np.sin(fase)
        filtro_integral += beta * error
        ajuste_fase = alpha * error + filtro_integral
        
        # Generar 38 kHz
        portadora_38k[i] = np.sin(2.0 * fase)
        
        fase += w_center + ajuste_fase
        if fase > 2.0 * np.pi: fase -= 2.0 * np.pi
        elif fase < 0.0: fase += 2.0 * np.pi
            
    return portadora_38k, fase, filtro_integral

class DemoduladorWBFM(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 10e6
        self.fft_size = 4096
        self.buffer_medicion = np.array([], dtype=np.complex128)
        
        # Filtros (Kernels FIR = Respuestas al Impulso h[n])
        self.bb_lpf_kernel = None
        self.mpx_lpf_kernel = None
        self.stereo_lpr_kernel = None 
        self.stereo_pilot_kernel = None 
        self.stereo_lmr_kernel = None 
        self.stereo_audio_lpf_kernel = None 
        
        # Estados PLL
        self.pll_fase = 0.0
        self.pll_filtro_integral = 0.0

    @property
    def id(self): return "wbfm"

    @property
    def nombre_mostrar(self): return "WBFM (Modo Medición Precisa)"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.buffer_medicion = np.array([], dtype=np.complex128)
        
        hw_sr = int(self.sample_rate)
        self.factor_bajada = int(hw_sr / 300000)
        self.nueva_fs = hw_sr / self.factor_bajada
        freq_nyq = self.nueva_fs / 2.0
        
        # Diseñamos las Respuestas al Impulso (h[n])
        # Usamos número impar de taps para simetría perfecta y fase lineal estricta
        self.bb_lpf_kernel = firwin(201, 200e3 / (hw_sr / 2.0))
        self.mpx_lpf_kernel = firwin(101, 80000 / freq_nyq)
        
        self.stereo_lpr_kernel = firwin(201, 15000 / freq_nyq)
        self.stereo_pilot_kernel = firwin(301, [18500 / freq_nyq, 19500 / freq_nyq], pass_zero=False)
        self.stereo_lmr_kernel = firwin(201, [23000 / freq_nyq, 53000 / freq_nyq], pass_zero=False)
        self.stereo_audio_lpf_kernel = firwin(201, 15000 / freq_nyq)
        
        self.pll_fase = 0.0
        self.pll_filtro_integral = 0.0

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        # === 1. ESPECTRO RF (RÁPIDO) ===
        fs_fft = self.fft_size
        PSD, rf_fft_chunk = None, None
        if len(muestras_iq) >= fs_fft:
            rf_fft_chunk = muestras_iq[:fs_fft] - np.mean(muestras_iq[:fs_fft])
            potencia = np.abs(np.fft.fftshift(np.fft.fft(rf_fft_chunk)))**2 / fs_fft
            PSD = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            centro = fs_fft // 2
            PSD[centro] = (PSD[centro - 1] + PSD[centro + 1]) / 2.0

        # === 2. ACUMULACIÓN DE MEDIO SEGUNDO ===
        self.buffer_medicion = np.append(self.buffer_medicion, muestras_iq)
        muestras_500ms = int(self.sample_rate * 0.5)
        
        if len(self.buffer_medicion) < muestras_500ms:
            return {'psd_rf': PSD, 'rf_chunk': rf_fft_chunk}
            
        # --- BLOQUE COMPLETO LISTO ---
        bloque_iq = self.buffer_medicion[:muestras_500ms]
        self.buffer_medicion = self.buffer_medicion[muestras_500ms:] 
        
        # === 3. DEMODULACIÓN ===
        
        # A. Filtrado de Banda Base: Convolución Compleja
        # y[n] = x[n] * h_bb[n]
        bloque_dc_block = bloque_iq - np.mean(bloque_iq)
        bb_filt = np.convolve(bloque_dc_block, self.bb_lpf_kernel, mode='same')
        
        # B. Downsampling
        bb_resample = resample_poly(bb_filt, up=1, down=self.factor_bajada)
        
        # C. Discriminador FM
        msj = np.angle(bb_resample[1:] * np.conjugate(bb_resample[:-1])) * (self.nueva_fs / (2*np.pi))
        demod_khz_raw = msj / 1000.0
        
        # D. Filtrado MPX: Convolución
        # mpx[n] = msj[n] * h_mpx[n]
        demod_khz_clean = np.convolve(demod_khz_raw, self.mpx_lpf_kernel, mode='same')
        
        # E. Espectro MPX Alta Resolución
        N_audio = len(demod_khz_clean)
        potencia_audio = np.abs(np.fft.fft(demod_khz_clean))**2 / N_audio
        mitad = N_audio // 2
        PSD_audio = 10.0 * np.log10(np.maximum(potencia_audio[:mitad], 1e-12))
        f_axis_audio = np.linspace(0, (self.nueva_fs/2)/1e3, mitad)
        
        # F. Extracción Estéreo 
        # L+R[n] = mpx[n] * h_lpr[n]
        senal_L_plus_R = np.convolve(demod_khz_clean, self.stereo_lpr_kernel, mode='same')
        
        # piloto[n] = mpx[n] * h_pilot[n]
        piloto_19k = np.convolve(demod_khz_clean, self.stereo_pilot_kernel, mode='same')
        
        # L-R_mod[n] = mpx[n] * h_lmr[n]
        lmr_modulado = np.convolve(demod_khz_clean, self.stereo_lmr_kernel, mode='same')
        
        # G. Regeneración de Portadora y Mezcla
        portadora_38k, self.pll_fase, self.pll_filtro_integral = correr_pll_38k(
            piloto_19k, self.nueva_fs, self.pll_fase, self.pll_filtro_integral
        )
        lmr_mezclado = lmr_modulado * portadora_38k * 2.0
        
        # L-R_demod[n] = mezclado[n] * h_lpf[n]
        senal_L_minus_R = np.convolve(lmr_mezclado, self.stereo_audio_lpf_kernel, mode='same')
        
        # H. Matriz Estéreo Pura
        canal_L = (senal_L_plus_R + senal_L_minus_R) / 2.0
        canal_R = (senal_L_plus_R - senal_L_minus_R) / 2.0
        
        # I. Métricas Absolutas
        fm_metrics = {
            'pico_max': np.max(demod_khz_clean),
            'pico_min': np.min(demod_khz_clean),
            'rms': np.sqrt(np.mean(demod_khz_clean**2)),
            'pico_rms': max(abs(np.max(demod_khz_clean)), abs(np.min(demod_khz_clean))) / np.sqrt(2),
            'dc_offset': np.mean(demod_khz_clean)
        }
        
        # J. Recorte del buffer para graficar estable
        inicio_plot = len(canal_L) // 2 
        muestras_10ms = int(self.nueva_fs * 0.01)
        fin_plot = inicio_plot + muestras_10ms
        
        t_axis_audio = np.linspace(0, 10, muestras_10ms)
        
        return {
            'psd_rf': PSD,
            'rf_chunk': rf_fft_chunk,
            'psd_mpx': PSD_audio,
            'f_axis_mpx': f_axis_audio,
            'mpx_time': demod_khz_clean[inicio_plot:fin_plot],
            'audio_time_L': canal_L[inicio_plot:fin_plot], 
            'audio_time_R': canal_R[inicio_plot:fin_plot],
            't_axis_audio': t_axis_audio,
            'audio_out': None, 
            'metricas': fm_metrics
        }