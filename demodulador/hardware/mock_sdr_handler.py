"""SDR sin hardware para analizar capturas IQ desde la interfaz."""

from .sdr_base import SDRBase


class MockSDRHandler(SDRBase):
    supports_live_rx = False

    def __init__(self, rx_callback=None):
        super().__init__(rx_callback)
        self.sample_rate = None
        self.center_freq = None

    @property
    def nombre(self):
        return "Reproducción de archivos (sin SDR)"

    def configurar(self, sample_rate, center_freq):
        self.set_sample_rate(sample_rate)
        self.set_freq(center_freq)

    def set_freq(self, freq_hz):
        self.center_freq = float(freq_hz)

    def set_sample_rate(self, sr_hz):
        self.sample_rate = float(sr_hz)

    def get_sample_rate(self):
        return self.sample_rate

    def get_center_frequency(self):
        return self.center_freq

    def set_gain(self, gain_db):
        pass

    def start_rx(self):
        pass

    def stop_rx(self):
        pass

    def close(self):
        pass
