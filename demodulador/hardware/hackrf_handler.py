import numpy as np
from python_hackrf import pyhackrf
from .sdr_base import SDRBase

class HackRFHandler(SDRBase):
    def __init__(self, rx_callback):
        super().__init__(rx_callback)
        pyhackrf.pyhackrf_init()
        self.sdr = pyhackrf.pyhackrf_open()
        
        # Le asignamos nuestro callback interno
        self.sdr.set_rx_callback(self._internal_rx_callback)

    @property
    def nombre(self):
        return "HackRF One"

    def _internal_rx_callback(self, device, buffer, buffer_length, valid_length):
        """
        Este es el callback que pide la librería en C. 
        Lo convertimos a math de Python y lo mandamos para arriba.
        """
        try:
            raw_data = np.array(buffer[:valid_length], dtype=np.int8)
            # Pasamos de I/Q intercalados a complex128
            c_samples = (raw_data[0::2] + 1j * raw_data[1::2]) / 128.0
            
            # Mandamos las muestras limpias a la aplicación (a tu futuro process_iq_samples)
            if self.is_running:
                self.rx_callback(c_samples)
        except Exception as e:
            print(f"Error crítico en rx_callback de HackRF: {e}")
            
        return 0 # pyhackrf exige devolver 0

    def configurar(self, sample_rate: float, center_freq: float):
        self.set_sample_rate(sample_rate)
        self.set_freq(center_freq)
        self.sdr.pyhackrf_set_lna_gain(8)
        self.sdr.pyhackrf_set_vga_gain(16)

    def set_freq(self, freq_hz: float):
        self.sdr.pyhackrf_set_freq(int(freq_hz))

    def set_sample_rate(self, sr_hz: float):
        self.sdr.pyhackrf_stop_rx()
        self.sdr.pyhackrf_set_sample_rate(int(sr_hz))
        bw = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(sr_hz * 0.75)
        self.sdr.pyhackrf_set_baseband_filter_bandwidth(bw)
        if self.is_running:
            self.sdr.pyhackrf_start_rx()

    def set_gain(self, gain_db: int):
        # Para la HackRF podríamos hacer que esto controle el LNA
        self.sdr.pyhackrf_set_lna_gain(gain_db)
    
    def set_vga_gain(self, gain_db: int):
        """Controla el Variable Gain Amplifier específico de la HackRF"""
        if self.sdr:
            self.sdr.pyhackrf_set_vga_gain(gain_db)

    def start_rx(self):
        self.is_running = True
        self.sdr.pyhackrf_start_rx()

    def stop_rx(self):
        self.is_running = False
        self.sdr.pyhackrf_stop_rx()

    def close(self):
        self.stop_rx()
        self.sdr.pyhackrf_close()
        pyhackrf.pyhackrf_exit()