import numpy as np
from .base import DemoduladorBase

class DemoduladorBTLE(DemoduladorBase):
    def __init__(self):
        super().__init__()
        self._id = 'btle'
        self._nombre_mostrar = 'BTLE (Bluetooth Low Energy)'
        self.sample_rate = 20e6
        self.fft_size = 2048
        
        self.buffer_len_s = 0.05
        self.buffer = np.array([], dtype=np.complex64)
        self.last_burst_metrics = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def nombre_mostrar(self) -> str:
        return self._nombre_mostrar

    def configurar(self, sample_rate: float, fft_size: int, bw_mhz: float = 1):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bw_mhz = bw_mhz
        self.buffer = np.array([], dtype=np.complex64)

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        self.buffer = np.concatenate((self.buffer, muestras_iq))
        
        target_len = int(self.sample_rate * self.buffer_len_s)
        resultados = {}
        
        if len(self.buffer) >= target_len:
            iq_samples = self.buffer[:target_len]
            overlap = int(self.sample_rate * 2e-3)
            self.buffer = self.buffer[target_len - overlap:]
            
            # 1. Burst Detection
            power = np.abs(iq_samples)**2
            window_size = int(self.sample_rate * 5e-6)
            if window_size == 0: window_size = 1
            window = np.ones(window_size) / window_size
            smoothed_power = np.convolve(power, window, mode='same')
            
            p_min = np.min(smoothed_power)
            p_max = np.max(smoothed_power)
            threshold = p_min + (p_max - p_min) * 0.2
            
            is_active = smoothed_power > threshold
            edges = np.diff(is_active.astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            
            if len(is_active) > 0 and is_active[0]: starts = np.insert(starts, 0, 0)
            if len(is_active) > 0 and is_active[-1]: ends = np.append(ends, len(iq_samples)-1)
            
            min_len = int(self.sample_rate * 20e-6)
            valid_bursts = []
            for s, e in zip(starts, ends):
                if (e - s) > min_len and s > 0 and e < len(iq_samples) - 1:
                    valid_bursts.append((s, e))
            
            # Determine the region to extract for analysis
            extract_start = None
            extract_end = None
            
            if len(valid_bursts) > 0:
                first_burst_start, first_burst_end = valid_bursts[0]
                margin = int(self.sample_rate * 10e-6)
                extract_start = max(0, first_burst_start - margin)
                extract_end = min(len(iq_samples), first_burst_end + margin)
            else:
                # Fallback for continuous signals (e.g., 10101010 test pattern).
                # When the signal is active most of the time, there are no clean
                # burst edges and the s > 0 boundary condition rejects everything.
                active_ratio = np.mean(is_active)
                if active_ratio > 0.8:
                    max_display_samples = int(self.sample_rate * 500e-6)  # 500 µs window
                    center = len(iq_samples) // 2
                    half = min(max_display_samples // 2, center)
                    extract_start = center - half
                    extract_end = center + half
            
            if extract_start is not None:
                burst_samples = iq_samples[extract_start:extract_end]
                burst_time_us = np.arange(len(burst_samples)) / self.sample_rate * 1e6
                
                # 2. Power vs Time
                power_mw_burst = np.abs(burst_samples)**2
                power_dbm = 10 * np.log10(power_mw_burst + 1e-12)
                
                # 3. Frequency Deviation
                phase = np.unwrap(np.angle(burst_samples))
                freq_dev_hz = np.diff(phase) / (2 * np.pi) * self.sample_rate
                freq_dev_hz = np.append(freq_dev_hz, freq_dev_hz[-1])
                freq_dev_khz = freq_dev_hz / 1000.0
                
                # 4. Spectrum ACP
                N_b = len(burst_samples)
                fft_vals = np.fft.fftshift(np.fft.fft(burst_samples)) / N_b
                power_spectrum_b = np.abs(fft_vals)**2
                freqs_b = np.fft.fftshift(np.fft.fftfreq(N_b, 1/self.sample_rate))
                
                channel_bandwidth = getattr(self, 'bw_mhz', 1) * 1e6
                offsets_mhz = np.arange(-10, 11)
                channel_offsets_ch = offsets_mhz / 2.0
                channel_power_dbm = []
                for offset_mhz in offsets_mhz:
                    center_f = offset_mhz * 1e6
                    idx = np.where((freqs_b >= center_f - channel_bandwidth/2) & (freqs_b <= center_f + channel_bandwidth/2))[0]
                    if len(idx) > 0:
                        pwr_mw = np.sum(power_spectrum_b[idx])
                        pwr_dbm = 10 * np.log10(pwr_mw + 1e-12)
                    else:
                        pwr_dbm = -100
                    channel_power_dbm.append(float(pwr_dbm))
                
                self.last_burst_metrics = {
                    'burst_time_us': burst_time_us,
                    'power_dbm': power_dbm,
                    'freq_dev_khz': freq_dev_khz,
                    'acp_channels': channel_offsets_ch,
                    'acp_power_dbm': np.array(channel_power_dbm)
                }

        fft_data = np.fft.fftshift(np.fft.fft(muestras_iq, n=self.fft_size))
        psd = 10 * np.log10(np.abs(fft_data)**2 + 1e-12)
        
        resultados['psd_rf'] = psd
        resultados['rf_chunk'] = np.array([])
        if self.last_burst_metrics is not None:
            resultados['metricas'] = {'btle_metrics': self.last_burst_metrics}
            
        return resultados
