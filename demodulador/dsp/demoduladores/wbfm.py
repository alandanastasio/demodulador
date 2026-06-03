import numpy as np
import threading
from scipy.signal import resample_poly, firwin, fftconvolve
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
        
        # Generar 38 kHz (Fase geométrica corregida)
        portadora_38k[i] = -np.sin(2.0 * fase)
        
        fase += w_center + ajuste_fase
        if fase > 2.0 * np.pi: fase -= 2.0 * np.pi
        elif fase < 0.0: fase += 2.0 * np.pi
            
    return portadora_38k, fase, filtro_integral

class DemoduladorWBFM(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 10e6
        self.fft_size = 4096
        self.buffer_medicion = []
        self.muestras_acumuladas = 0
        
        # Filtros (Kernels FIR)
        self.bb_lpf_kernel = None
        self.mpx_lpf_kernel = None
        self.stereo_lpr_kernel = None 
        self.stereo_pilot_kernel = None 
        self.stereo_lmr_kernel = None 
        self.stereo_audio_lpf_kernel = None 
        
        # Estados PLL
        self.pll_fase = 0.0
        self.pll_filtro_integral = 0.0
        
        # ---  Control de Hilos (Asincronismo) ---
        self.is_processing = False
        self.last_heavy_results = {}
        self.nuevos_datos_listos = False

    @property
    def id(self): return "wbfm"

    @property
    def nombre_mostrar(self): return "WBFM (Modo Medición Precisa)"

    def configurar(self, sample_rate: float, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.buffer_medicion = []
        self.muestras_acumuladas = 0
        
        hw_sr = int(self.sample_rate)
        self.factor_bajada = int(hw_sr / 300000)
        self.nueva_fs = hw_sr / self.factor_bajada
        freq_nyq = self.nueva_fs / 2.0
        
        # Filtros de Grado Laboratorio (501 taps y Blackman-Harris)
        self.bb_lpf_kernel = firwin(501, 200e3 / (hw_sr / 2.0), window='blackmanharris')
        self.mpx_lpf_kernel = firwin(501, 80000 / freq_nyq, window='blackmanharris')
        self.stereo_lpr_kernel = firwin(501, 15000 / freq_nyq, window='blackmanharris')
        self.stereo_pilot_kernel = firwin(501, [18500 / freq_nyq, 19500 / freq_nyq], pass_zero=False, window='blackmanharris')
        self.stereo_lmr_kernel = firwin(501, [23000 / freq_nyq, 53000 / freq_nyq], pass_zero=False, window='blackmanharris')
        self.stereo_audio_lpf_kernel = firwin(501, 15000 / freq_nyq, window='blackmanharris')
        
        self.pll_fase = 0.0
        self.pll_filtro_integral = 0.0
        self.is_processing = False
        self.last_heavy_results = {}

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        """ Recepcionista ultrarrápida. Cero matemáticas. """
        self.buffer_medicion.append(muestras_iq)
        self.muestras_acumuladas += len(muestras_iq)
        
        muestras_500ms = int(self.sample_rate * 0.5) 
        
        if self.muestras_acumuladas >= muestras_500ms:
            if not self.is_processing:
                bloque_iq = np.concatenate(self.buffer_medicion)[:muestras_500ms]
                self.is_processing = True
                threading.Thread(target=self._procesar_fondo, args=(bloque_iq,)).start()
            
            self.buffer_medicion = []
            self.muestras_acumuladas = 0
        
        # --- Sincronización perfecta de todos los gráficos ---
        if self.nuevos_datos_listos:
            self.nuevos_datos_listos = False
            return self.last_heavy_results
            
        # Si el hilo sigue trabajando, devolvemos None. 
        # La GUI (main.py) ignorará el frame y descansará la CPU.
        return None 

    def _procesar_fondo(self, bloque_iq):
        """ Hilo de fondo: Trabaja pesado una vez cada 500ms """
        try:
            # === 1. ESPECTRO RF (Movido al hilo para ahorrar recursos) ===
            fs_fft = self.fft_size
            rf_fft_chunk = bloque_iq[:fs_fft] - np.mean(bloque_iq[:fs_fft])
            ventana_rf = np.hanning(fs_fft)
            chunk_ventaneado = rf_fft_chunk * ventana_rf
            
            potencia = np.abs(np.fft.fftshift(np.fft.fft(chunk_ventaneado)))**2 / fs_fft
            PSD_rf = 10.0 * np.log10(np.maximum(potencia, 1e-12))
            centro = fs_fft // 2
            PSD_rf[centro] = (PSD_rf[centro - 1] + PSD_rf[centro + 1]) / 2.0

            # === 2. DEMODULACIÓN BANDA BASE ===
            bloque_dc_block = bloque_iq - np.mean(bloque_iq)
            bb_filt = fftconvolve(bloque_dc_block, self.bb_lpf_kernel, mode='same')
            
            bb_resample = resample_poly(bb_filt, up=1, down=self.factor_bajada)
            
            msj = np.angle(bb_resample[1:] * np.conjugate(bb_resample[:-1])) * (self.nueva_fs / (2*np.pi))
            demod_khz_raw = msj / 1000.0
            
            demod_khz_clean = fftconvolve(demod_khz_raw, self.mpx_lpf_kernel, mode='same')
            
            # === 3. ESPECTRO MPX (Alta Resolución Restaurada) ===
            N_audio = len(demod_khz_clean)
            
            # Quitamos la continua y ventaneamos
            mpx_sin_dc = demod_khz_clean - np.mean(demod_khz_clean) 
            ventana_mpx = np.hanning(N_audio)
            mpx_ventaneado = mpx_sin_dc * ventana_mpx
            
            potencia_audio = np.abs(np.fft.fft(mpx_ventaneado))**2 / N_audio
            mitad = N_audio // 2
            PSD_audio = 10.0 * np.log10(np.maximum(potencia_audio[:mitad], 1e-12))
            f_axis_audio = np.linspace(0, (self.nueva_fs/2)/1e3, mitad)
            
            # === 4. EXTRACCIÓN ESTÉREO ===
            senal_L_plus_R = fftconvolve(demod_khz_clean, self.stereo_lpr_kernel, mode='same')
            piloto_19k = fftconvolve(demod_khz_clean, self.stereo_pilot_kernel, mode='same')
            lmr_modulado = fftconvolve(demod_khz_clean, self.stereo_lmr_kernel, mode='same')
            
            portadora_38k, pll_f_nuevo, pll_i_nuevo = correr_pll_38k(
                piloto_19k, self.nueva_fs, self.pll_fase, self.pll_filtro_integral
            )
            self.pll_fase = pll_f_nuevo
            self.pll_filtro_integral = pll_i_nuevo
            
            lmr_mezclado = lmr_modulado * portadora_38k * 2.0
            senal_L_minus_R = fftconvolve(lmr_mezclado, self.stereo_audio_lpf_kernel, mode='same')
            
            canal_L = (senal_L_plus_R + senal_L_minus_R) / 2.0
            canal_R = (senal_L_plus_R - senal_L_minus_R) / 2.0
            
            # === 5. MÉTRICAS Y PREPARACIÓN FINAL ===
            inicio_plot = len(canal_L) // 2 
            muestras_10ms = int(self.nueva_fs * 0.01)
            fin_plot = inicio_plot + muestras_10ms
            
            mpx_medicion = demod_khz_clean[inicio_plot:fin_plot]
            L_medicion = canal_L[inicio_plot:fin_plot]
            R_medicion = canal_R[inicio_plot:fin_plot]
            
            rms_L = np.sqrt(np.mean(L_medicion**2))
            rms_R = np.sqrt(np.mean(R_medicion**2))

            fm_metrics = {
                'pico_max': np.max(mpx_medicion),
                'pico_min': np.min(mpx_medicion),
                'rms': np.sqrt(np.mean(mpx_medicion**2)),
                'pico_rms': max(abs(np.max(mpx_medicion)), abs(np.min(mpx_medicion))) / np.sqrt(2),
                'dc_offset': np.mean(mpx_medicion),
                'rms_L': rms_L,
                'rms_R': rms_R
            }
            
            t_axis_audio = np.linspace(0, 10, muestras_10ms)
            
            # Enviamos el RF y el MPX todo en un mismo paquete sincronizado
            self.last_heavy_results = {
                'psd_rf': PSD_rf,
                'rf_chunk': rf_fft_chunk,
                'psd_mpx': PSD_audio,
                'f_axis_mpx': f_axis_audio,
                'mpx_time': mpx_medicion,
                'audio_time_L': L_medicion, 
                'audio_time_R': R_medicion,
                't_axis_audio': t_axis_audio,
                'metricas': fm_metrics
            }
            
            self.nuevos_datos_listos = True 
            
        finally:
            self.is_processing = False