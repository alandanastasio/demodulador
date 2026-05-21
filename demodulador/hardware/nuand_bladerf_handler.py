import threading
import time
import numpy as np
import bladerf
from .sdr_base import SDRBase

class BladeRFHandler(SDRBase):
    def __init__(self, rx_callback):
        super().__init__(rx_callback)
        self.sdr = bladerf.BladeRF()
        self.rx_ch = self.sdr.Channel(bladerf.CHANNEL_RX(0))
        self._thread = None
        
        # Configuración del buffer de altísima velocidad
        self.sdr.sync_config(
            layout=bladerf._bladerf.ChannelLayout.RX_X1,
            fmt=bladerf._bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=32768,
            num_transfers=8,
            stream_timeout=3500
        )

    @property
    def nombre(self):
        return "Nuand bladeRF x40"

    def configurar(self, sample_rate: float, center_freq: float):
        self.set_sample_rate(sample_rate)
        self.set_freq(center_freq)
        self.set_gain(0) # Ganancia global inicial conservadora

    def set_freq(self, freq_hz: float):
        self.rx_ch.frequency = int(freq_hz)

    def set_sample_rate(self, sr_hz: float):
        self.rx_ch.sample_rate = int(sr_hz)
        self.rx_ch.bandwidth = int(sr_hz / 2) # Filtro antialiasing a la mitad

    def set_gain(self, gain_db: int):
        self.rx_ch.gain = int(gain_db)

    def _rx_worker(self):
        """
        Hilo dedicado a succionar datos del USB mediante sync_rx.
        """
        bytes_per_sample = 4 # SC16_Q11 = 2 enteros de 16 bits (I y Q)
        buf_size = 32768
        buf = bytearray(buf_size * bytes_per_sample)
        
        self.rx_ch.enable = True
        
        while self.is_running:
            try:
                self.sdr.sync_rx(buf, buf_size)
                
                # Conversión optimizada de bytes a int16 y luego a complex128
                data = np.frombuffer(buf, dtype=np.int16)
                c_samples = (data[0::2] + 1j * data[1::2]) / 2048.0 
                
                self.rx_callback(c_samples)
                
            except Exception as e:
                print(f"Error en rx_worker de bladeRF: {e}")
                break
                
        self.rx_ch.enable = False

    def start_rx(self):
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._rx_worker)
            self._thread.daemon = True
            self._thread.start()

    def stop_rx(self):
        if self.is_running:
            self.is_running = False
            # Damos tiempo a que el while del worker frene y libere el USB
            time.sleep(0.1) 
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)

    def close(self):
        self.stop_rx()
        self.sdr.close()