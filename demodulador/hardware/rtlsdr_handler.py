import threading
import logging
import numpy as np
from rtlsdr import RtlSdr
from .sdr_base import SDRBase

class RtlSdrHandler(SDRBase):
    def __init__(self, rx_callback):
        super().__init__(rx_callback)
        self.sdr = RtlSdr()
        self._thread = None
        self.sdr.gain = 'auto'

    @property
    def nombre(self):
        return "RTL-SDR"

    def get_sample_rate(self):
        return float(self.sdr.sample_rate)

    def get_center_frequency(self):
        return float(self.sdr.center_freq)

    def _async_callback(self, samples, context):
        """
        La RTL-SDR ya nos entrega las muestras como np.complex128,
        así que las pasamos directo a nuestro callback principal.
        """
        if self.is_running:
            self.rx_callback(samples)

    def configurar(self, sample_rate: float, center_freq: float):
        # La RTL no se banca 10 MHz, forzamos un máximo seguro si es necesario
        sr = min(sample_rate, 2.88e6) 
        self.set_sample_rate(sr)
        self.set_freq(center_freq)

    def set_freq(self, freq_hz: float):
        self.sdr.center_freq = freq_hz

    def set_sample_rate(self, sr_hz: float):
        self.sdr.sample_rate = sr_hz

    def set_gain(self, gain_db: int):
        # La RTL maneja ganancia en string 'auto' o valores discretos
        try:
            self.sdr.gain = gain_db
        except:
            self.sdr.gain = 'auto'

    def start_rx(self):
        if not self.is_running:
            self.is_running = True
            # Iniciamos la lectura asincrónica en un hilo separado
            self._thread = threading.Thread(
                target=self._rx_worker,
            )
            self._thread.daemon = True
            self._thread.start()

    def _rx_worker(self):
        try:
            self.sdr.read_samples_async(self._async_callback, 32768)
        except Exception:
            logging.exception("Falló el flujo IQ de RTL-SDR")
        finally:
            if self.is_running:
                self.is_running = False
                if self.rx_callback is not None:
                    self.rx_callback(None)

    def stop_rx(self):
        if self.is_running:
            self.is_running = False
            self.sdr.cancel_read_async()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    raise RuntimeError("La lectura RTL-SDR no se detuvo antes de reconfigurar el sintonizador.")
            self._thread = None

    def close(self):
        self.stop_rx()
        self.sdr.close()
