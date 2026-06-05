import numpy as np
import queue
import sounddevice as sd

class AudioManager:
    def __init__(self, main_window, state):
        self.ui = main_window
        self.state = state
        
        # Inicializamos las variables de audio directamente en el manager
        self.state['play_audio'] = False
        self.state['play_audio_L'] = False
        self.state['play_audio_R'] = False
        self.state['audio_queue'] = queue.Queue(maxsize=20)
        self.state['audio_buffer'] = np.zeros((0, 2), dtype=np.float32)
        
        self.audio_stream = None

    def enqueue_audio(self, audio_data):
        """Recibe las muestras del DSP y las mete en la cola si el audio está activo"""
        if self.state.get('play_audio', False) and audio_data is not None:
            if not self.state['audio_queue'].full():
                self.state['audio_queue'].put(audio_data)

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
            self.state['audio_buffer'] = np.zeros((0, 2), dtype=np.float32)
            while not self.state['audio_queue'].empty():
                self.state['audio_queue'].get()

            def audio_callback(outdata, frames, time, status):
                try:
                    while len(self.state['audio_buffer']) < frames:
                        new_data = self.state['audio_queue'].get_nowait()
                        # Si recibimos mono por accidente, lo duplicamos a estéreo
                        if new_data.ndim == 1:
                            new_data = np.column_stack((new_data, new_data))
                        self.state['audio_buffer'] = np.vstack((self.state['audio_buffer'], new_data))
                except queue.Empty:
                    pass
                
                if len(self.state['audio_buffer']) >= frames:
                    chunk = self.state['audio_buffer'][:frames].copy()
                    self.state['audio_buffer'] = self.state['audio_buffer'][frames:]
                    
                    if self.state.get('play_audio_L', False):
                        outdata[:, 0] = chunk[:, 0]
                    else:
                        outdata[:, 0] = 0.0
                        
                    if self.state.get('play_audio_R', False):
                        outdata[:, 1] = chunk[:, 1]
                    else:
                        outdata[:, 1] = 0.0
                        
                else:
                    outdata.fill(0.0)
                    self.state['audio_buffer'] = np.zeros((0, 2), dtype=np.float32)

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