import queue
import sounddevice as sd
from threading import Lock

class AudioManager:
    def __init__(self, main_window, state):
        self.ui = main_window
        self.state = state
        
        # Inicializamos las variables de audio directamente en el manager
        self.state['play_audio'] = False
        self.state['play_audio_L'] = False
        self.state['play_audio_R'] = False
        # Dos bloques de reserva absorben ráfagas de carga del DSP y la UI.
        self.state['audio_queue'] = queue.Queue(maxsize=6)
        self._audio_chunk = None
        self._audio_offset = 0
        self._primed = False
        self._audio_lock = Lock()
        self.underrun_events = 0
        self.underrun_frames = 0
        self.device_underruns = 0
        self.dropped_blocks = 0
        
        self.audio_stream = None

    def enqueue_audio(self, audio_data):
        """Recibe las muestras del DSP y las mete en la cola si el audio está activo"""
        if self.state.get('play_audio', False) and audio_data is not None and len(audio_data):
            try:
                self.state['audio_queue'].put_nowait(audio_data)
            except queue.Full:
                # En vivo importa más la muestra reciente que reproducir
                # segundos atrasados. Descartamos el bloque más antiguo.
                try:
                    self.state['audio_queue'].get_nowait()
                    self.dropped_blocks += 1
                except queue.Empty:
                    pass
                try:
                    self.state['audio_queue'].put_nowait(audio_data)
                except queue.Full:
                    pass

    def clear_audio(self):
        with self._audio_lock:
            self._audio_chunk = None
            self._audio_offset = 0
            self._primed = False
        while True:
            try:
                self.state['audio_queue'].get_nowait()
            except queue.Empty:
                break

    def deactivate(self):
        self.ui.audio_l_btn.setChecked(False)
        self.ui.audio_r_btn.setChecked(False)
        self.toggle_audio()

    def toggle_audio(self):
        """Enciende o apaga los canales estéreo y maneja el hardware de sonido"""
        play_l = self.ui.audio_l_btn.isChecked()
        play_r = self.ui.audio_r_btn.isChecked()
        
        # Cambiamos los colores (Cyan para L, Magenta para R)
        self.ui.audio_l_btn.setStyleSheet("background-color: #00FFFF; color: black; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #00CCCC;" if play_l else "background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")
        self.ui.audio_r_btn.setStyleSheet("background-color: #FF00FF; color: black; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #CC00CC;" if play_r else "background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")

        self.state['play_audio'] = play_l or play_r
        self.state['play_audio_L'] = play_l
        self.state['play_audio_R'] = play_r

        # Si hay alguno encendido y el stream no está corriendo, lo iniciamos
        if self.state['play_audio'] and (self.audio_stream is None or not self.audio_stream.active):
            # Limpiamos buffers
            self.clear_audio()
            self.underrun_events = 0
            self.underrun_frames = 0
            self.device_underruns = 0
            self.dropped_blocks = 0

            def audio_callback(outdata, frames, time, status):
                outdata.fill(0.0)
                with self._audio_lock:
                    if getattr(status, 'output_underflow', False):
                        self.device_underruns += 1
                    # Dos bloques de reserva absorben la variación de tiempo
                    # entre callbacks de IQ y ejecución del worker DSP.
                    if not self._primed:
                        if self.state['audio_queue'].qsize() < 3:
                            return
                        self._primed = True
                    written = 0
                    while written < frames:
                        if self._audio_chunk is None or self._audio_offset >= len(self._audio_chunk):
                            try:
                                self._audio_chunk = self.state['audio_queue'].get_nowait()
                                self._audio_offset = 0
                            except queue.Empty:
                                self._primed = False
                                self.underrun_events += 1
                                self.underrun_frames += frames - written
                                break
                        chunk = self._audio_chunk
                        count = min(frames - written, len(chunk) - self._audio_offset)
                        source = chunk[self._audio_offset:self._audio_offset + count]
                        if self.state.get('play_audio_L', False):
                            outdata[written:written + count, 0] = (
                                source if source.ndim == 1 else source[:, 0])
                        if self.state.get('play_audio_R', False):
                            outdata[written:written + count, 1] = (
                                source if source.ndim == 1 else source[:, 1])
                        self._audio_offset += count
                        written += count

            # Iniciamos el stream 
            self.audio_stream = sd.OutputStream(
                samplerate=48000, 
                channels=2, 
                dtype='float32',
                callback=audio_callback
            )
            self.audio_stream.start()
            
        # Si apagamos ambos botones y el stream sigue corriendo, lo detenemos
        elif not self.state['play_audio'] and self.audio_stream is not None:
            self.stop_all()

    def stop_all(self):
        """Cierra el puerto de audio de forma segura"""
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        self.clear_audio()

    def flow_stats(self):
        return {
            'audio_underruns': self.underrun_events,
            'audio_underrun_ms': self.underrun_frames / 48,
            'device_underruns': self.device_underruns,
            'dropped_audio_blocks': self.dropped_blocks,
        }
