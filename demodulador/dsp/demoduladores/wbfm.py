import numpy as np
from scipy.signal import butter, lfilter, resample_poly, lfilter_zi, firwin
from .base import DemoduladorBase

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
        self.deemph_zi = None
        self.fm_buffer = np.array([], dtype=np.complex128)

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
            
        # === PASO 4: DEMODULACIÓN FM (Ángulo del producto conjugado) ===
        chunk_with_last = np.insert(bb_resample, 0, self.fm_last_iq)
        self.fm_last_iq = bb_resample[-1]
        msj = np.angle(chunk_with_last[1:] * np.conjugate(chunk_with_last[:-1])) * (nueva_fs / (2*np.pi))
        demod_khz_raw = msj / 1000.0
        
        # === FILTRO MPX ===
        if self.mpx_zi is None:
            self.mpx_zi = lfilter_zi(self.mpx_lpf_kernel, [1.0]) * demod_khz_raw[0]
        demod_khz_clean, self.mpx_zi = lfilter(self.mpx_lpf_kernel, [1.0], demod_khz_raw, zi=self.mpx_zi)
        
        # --- ESPECTRO MPX ---
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
        
        # --- PROCESAMIENTO FINAL DE AUDIO (DE-ÉNFASIS Y NORMALIZACIÓN) ---
        audio_48k = resample_poly(demod_khz_clean, 4, 25)
        b_aud, a_aud = butter(1, 2122 / (48000/2), btype='low')
        if self.deemph_zi is None:
            self.deemph_zi = lfilter_zi(b_aud, a_aud) * audio_48k[0]
        audio_filtrado, self.deemph_zi = lfilter(b_aud, a_aud, audio_48k, zi=self.deemph_zi)
        
        audio_filtrado = audio_filtrado - np.mean(audio_filtrado)
        max_val = np.max(np.abs(audio_filtrado))
        audio_norm = np.float32(audio_filtrado / max(max_val, 15.0)) if max_val > 0 else np.float32(audio_filtrado)
        
        # Generación del snippet de tiempo para el osciloscopio
        muestras_10ms = int(nueva_fs * 0.01)
        audio_time_snippet = demod_khz_clean[:muestras_10ms]
        t_axis_audio = np.linspace(0, 10, len(audio_time_snippet))
        
        # Empaquetamos todo y lo escupimos
        return {
            'psd_rf': PSD,
            'rf_chunk': rf_fft_chunk,
            'psd_mpx': PSD_audio,
            'f_axis_mpx': f_axis_audio,
            'audio_time': audio_time_snippet,
            't_axis_audio': t_axis_audio,
            'audio_out': audio_norm,
            'metricas': fm_metrics
        }