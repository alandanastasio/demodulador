import threading
import time
import numpy as np
import uhd
from .sdr_base import SDRBase

class USRPB200Handler(SDRBase):
    def __init__(self, rx_callback):
        super().__init__(rx_callback)
        
        # Inicializamos el USRP. Le pasamos type=b200 para asegurarnos de agarrar la placa correcta.
        try:
            self.usrp = uhd.usrp.MultiUSRP("type=b200")
        except RuntimeError as e:
            print("⏳ [B200] Cargando firmware y esperando renumeración USB...")
            time.sleep(3.5) # Le damos tiempo a Ubuntu para que monte el nuevo USB
            self.usrp = uhd.usrp.MultiUSRP("type=b200")
        self._thread = None
        self.muestras_por_bloque = 32768
        
        # Configuramos el streamer.
        # fc32: Host format (Float Complex 32 bits = np.complex64)
        # sc16: Wire format (Short Complex 16 bits) -> Optimiza el ancho de banda USB
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self.streamer = self.usrp.get_rx_stream(st_args)

    @property
    def nombre(self) -> str:
        return "Ettus USRP B200"

    def configurar(self, sample_rate: float, center_freq: float):
        self.set_sample_rate(sample_rate)
        self.set_freq(center_freq)
        self.set_gain(40) # Arrancamos con una ganancia media razonable para la B200

    def set_muestras_por_bloque(self, muestras: int):
        self.muestras_por_bloque = int(muestras)

    def set_freq(self, freq_hz: float):
        tune_req = uhd.types.TuneRequest(freq_hz)
        self.usrp.set_rx_freq(tune_req, 0)

    def set_sample_rate(self, sr_hz: float):
        self.usrp.set_rx_rate(sr_hz, 0)
        self.usrp.set_rx_bandwidth(sr_hz, 0)

    def set_gain(self, gain_db: int):
        # La B200 normalmente tiene un rango de 0 a 73 o 76 dB.
        self.usrp.set_rx_gain(gain_db, 0)

    def _rx_worker(self):
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.streamer.issue_stream_cmd(stream_cmd)
        
        metadata = uhd.types.RXMetadata()
        
        # ✅ RESERVAMOS MEMORIA UNA SOLA VEZ AFUERA DEL BUCLE
        chunk_size = self.muestras_por_bloque
        recv_buffer = np.zeros(chunk_size, dtype=np.complex64)
        
        while self.is_running:
            try:
                # Si el tamaño cambia dinámicamente desde la UI, redimensionamos el buffer
                if chunk_size != self.muestras_por_bloque:
                    chunk_size = self.muestras_por_bloque
                    recv_buffer = np.zeros(chunk_size, dtype=np.complex64)

                samps_received = 0
                
                while samps_received < chunk_size and self.is_running:
                    # Sobrescribimos el buffer existente en lugar de crear uno nuevo
                    samps = self.streamer.recv(recv_buffer[samps_received:chunk_size], metadata)
                    
                    if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                        if metadata.error_code != uhd.types.RXMetadataErrorCode.timeout:
                            # Ignoramos timeouts cortos, pero alertamos de otras cosas (como overflows)
                            pass 
                            
                    samps_received += samps
                
                if samps_received == chunk_size and self.is_running:
                    # Mandamos una copia casteada al callback
                    self.rx_callback(recv_buffer.astype(np.complex128))
                    
            except Exception as e:
                print(f"Error en rx_worker de USRP: {e}")
                break
                
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.streamer.issue_stream_cmd(stop_cmd)

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
        # En la API de Python de UHD, la liberación de recursos es manejada por 
        # el garbage collector, pero eliminamos la referencia para forzarla.
        self.streamer = None
        self.usrp = None