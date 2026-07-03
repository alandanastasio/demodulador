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
        self.muestras_por_bloque = 8192  # valor por defecto hasta que main lo configure
             

    @property
    def nombre(self):
        return "Nuand bladeRF x40"

    def configurar(self, sample_rate: float, center_freq: float):
        self.set_sample_rate(sample_rate)
        self.set_freq(center_freq)
        self.set_gain(0) 

    def set_muestras_por_bloque(self, muestras: int):
        pot2 = 1
        while pot2 < int(muestras):
            pot2 *= 2
        self.muestras_por_bloque = pot2
        print(f"[bladeRF] muestras_por_bloque={pot2}")

    def set_freq(self, freq_hz: float):
        self.rx_ch.frequency = int(freq_hz)

    def set_sample_rate(self, sr_hz: float):
        self.rx_ch.sample_rate = int(sr_hz)
        self.rx_ch.bandwidth = int(sr_hz)

    def set_gain(self, gain_db: int):
        self.rx_ch.gain = int(gain_db)

    def _rx_worker(self):
        bytes_per_sample = 4
    
        # Primero habilitar el canal
        self.rx_ch.enable = True
        
        # Luego (re)configurar el sync
        self.sdr.sync_config(
            layout=bladerf._bladerf.ChannelLayout.RX_X1,
            fmt=bladerf._bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=self.muestras_por_bloque,
            num_transfers=8,
            stream_timeout=3500
        )
        
        tamaño_actual = self.muestras_por_bloque
        buf = bytearray(tamaño_actual * bytes_per_sample)
        
        while self.is_running:
            try:
                # Solo reasignamos si el usuario cambió el tamaño desde la UI
                if tamaño_actual != self.muestras_por_bloque:
                    tamaño_actual = self.muestras_por_bloque
                    buf = bytearray(tamaño_actual * bytes_per_sample)
                
                # sync_rx llena el buffer in-place (bloqueante)
                self.sdr.sync_rx(buf, tamaño_actual)
                
                # 2. OPTIMIZACIÓN MATEMÁTICA EXTREMA
                # Vista int16 directa del buffer (cero copias)
                data = np.frombuffer(buf, dtype=np.int16)
                
                # Casteo a float32 y escalado en un solo paso
                data_f = data.astype(np.float32) * (1.0 / 2048.0)
                
                # Truco de magia: Como data_f es [I, Q, I, Q] en memoria float32,
                # .view(np.complex64) lo interpreta directamente como complejo 
                # emparejando de a dos, sin hacer matemática ni usar RAM extra.
                c_samples = data_f.view(np.complex64)
                
                # Opcional: remover el DC offset si ves que el S&C sigue fallando
                c_samples = c_samples - np.mean(c_samples)
                
                self.rx_callback(c_samples.astype(np.complex128))
                
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