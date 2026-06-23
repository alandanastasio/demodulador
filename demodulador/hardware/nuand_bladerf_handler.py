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
        
        # Agregamos una variable para controlar el tamaño de la captura desde afuera
        self.muestras_por_bloque = 8192 
        
        self.sdr.sync_config(
            layout=bladerf._bladerf.ChannelLayout.RX_X1,
            fmt=bladerf._bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=8192,
            num_transfers=8,
            stream_timeout=3500
        )

    @property
    def nombre(self):
        return "Nuand bladeRF x40"

    def configurar(self, sample_rate: float, center_freq: float):
        self.set_sample_rate(sample_rate)
        self.set_freq(center_freq)
        self.set_gain(0) 

    # --- NUEVO MÉTODO PARA CAMBIAR EL TAMAÑO ---
    def set_muestras_por_bloque(self, muestras: int):
        self.muestras_por_bloque = int(muestras)

    def set_freq(self, freq_hz: float):
        self.rx_ch.frequency = int(freq_hz)

    def set_sample_rate(self, sr_hz: float):
        self.rx_ch.sample_rate = int(sr_hz)
        self.rx_ch.bandwidth = int(sr_hz)

    def set_gain(self, gain_db: int):
        self.rx_ch.gain = int(gain_db)

    def _rx_worker(self):
        bytes_per_sample = 4 
        self.rx_ch.enable = True
        
        while self.is_running:
            try:
                # Tomamos el valor actual por si cambió en el medio de la ejecución
                tamaño_actual = self.muestras_por_bloque
                buf = bytearray(tamaño_actual * bytes_per_sample)
                
                # sync_rx se clava acá hasta que llena el buffer completo que le pediste
                self.sdr.sync_rx(buf, tamaño_actual)
                
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
            time.sleep(0.1) 
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)

    def close(self):
        self.stop_rx()
        self.sdr.close()