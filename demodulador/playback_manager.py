import numpy as np
import datetime
import pyqtgraph as pg
from PyQt6.QtWidgets import QFileDialog, QApplication
from PyQt6.QtCore import QTimer

class PlaybackManager:
    def __init__(self, main_window, app_state, emitter):
        self.ui = main_window
        self.state = app_state
        self.emitter = emitter
        
        # Variables internas de reproducción
        self.playback_data = None
        self.playback_index = 0
        self.is_looping = False
        
        # Timer nativo de PyQt para controlar la velocidad (FPS)
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.playback_step)

    def toggle_recording(self):
        if not self.state['is_recording']:
            self.state['is_recording'] = True
            self.state['recorded_samples'] = [] # Limpiamos memoria RAM
            
            self.ui.record_action.setText("⏹ Detener y Guardar")
            self.ui.rec_play_btn.setStyleSheet("background-color: #8b0000; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
            print("Grabación de muestras IQ iniciada...")
            
            # Bloqueamos controles para evitar cambios durante la grabación
            self.ui.freq_input.setEnabled(False)
            self.ui.sr_combo.setEnabled(False)
            self.ui.fft_combo.setEnabled(False)
        else:
            self.state['is_recording'] = False
            
            # Cambiamos estado visual y FORZAMOS a la GUI a actualizarse
            self.ui.record_action.setText("⏳ Guardando...")
            self.ui.rec_play_btn.setText("⏳ Guardando...")
            self.ui.rec_play_btn.setStyleSheet("background-color: #ff8c00; color: black; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
            QApplication.processEvents() 

            if len(self.state['recorded_samples']) > 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"muestras_iq_{timestamp}.npz"
                
                # Unir todos los fragmentos interceptados en un solo array gigante
                todas_las_muestras = np.concatenate(self.state['recorded_samples'])
                
                np.savez(
                    filename,
                    raw_iq=todas_las_muestras,
                    center_freq=self.state['center_freq'],
                    sample_rate=self.state['sample_rate']
                )
                print(f"Grabación guardada exitosamente en: {filename}")
                print(f"Muestras totales grabadas: {len(todas_las_muestras)}")
                self.state['recorded_samples'] = [] # Liberamos la RAM

            # Restauramos apariencia normal
            self.ui.record_action.setText("🔴 Iniciar Grabación")
            self.ui.rec_play_btn.setText("Rec/Play")
            self.ui.rec_play_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
            
            self.ui.freq_input.setEnabled(True)
            self.ui.sr_combo.setEnabled(True)
            self.ui.fft_combo.setEnabled(True)

    def load_and_play(self, loop=False):
        filename, _ = QFileDialog.getOpenFileName(self.ui, "Seleccionar Grabación IQ", "", "Numpy Archives (*.npz)")
        if not filename:
            return

        # Apagamos la radio usando la interfaz modular (Arreglo del bug viejo)
        if hasattr(self.ui.radio, 'stop_rx'):
            self.ui.radio.stop_rx()
            
        print(f"Cargando archivo: {filename}...")
        QApplication.processEvents()

        try:
            data = np.load(filename)
            self.playback_data = data['raw_iq']
            cf = float(data['center_freq'])
            sr = float(data['sample_rate'])
        except Exception as e:
            print(f"Error al leer el archivo: {e}")
            if hasattr(self.ui.radio, 'start_rx'):
                self.ui.radio.start_rx()
            return

        self.is_looping = loop
        self.state['center_freq'] = cf
        self.state['sample_rate'] = sr
        
        # Actualizamos la caja de texto sin disparar la señal de radio
        self.ui.freq_input.blockSignals(True)
        self.ui.freq_input.setValue(cf / self.ui.current_freq_multiplier)
        self.ui.freq_input.blockSignals(False)
        self.ui.update_x_axis()

        # Bloquear y desbloquear controles
        self.ui.freq_input.setEnabled(False)
        self.ui.sr_combo.setEnabled(False)
        self.ui.fft_combo.setEnabled(False)
        self.ui.record_action.setEnabled(False)
        self.ui.play_action.setEnabled(False)
        self.ui.loop_action.setEnabled(False)
        self.ui.stop_play_action.setEnabled(True)
        
        if self.is_looping:
            self.ui.rec_play_btn.setText("🔁 Reproduciendo Loop...")
        else:
            self.ui.rec_play_btn.setText("▶ Reproduciendo...")
            
        self.ui.rec_play_btn.setStyleSheet("background-color: #004d99; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
        self.ui.freq_plot_curve.setPen(pg.mkPen(color="#FF8C00", width=1.5))

        self.playback_index = 0
        self.playback_timer.start(33) # Correr a ~30 FPS
        print("Reproducción iniciada.")

    def playback_step(self):
        # Calculamos cuántas muestras equivalen a ~33ms (nuestro timer a 30 FPS)
        chunk_size = int(self.state['sample_rate'] * 0.033) 

        # Verificamos si nos quedamos sin muestras para leer
        if self.playback_index + chunk_size > len(self.playback_data):
            if self.is_looping:
                self.playback_index = 0 
            else:
                self.stop_playback() 
                return

        # Extraemos el bloque de muestras continuas simulando el buffer del SDR
        chunk = self.playback_data[self.playback_index : self.playback_index + chunk_size].copy()
        
        # Avanzamos el puntero en el tiempo
        self.playback_index += chunk_size

        # Le inyectamos las muestras al puente principal de la aplicación.
        # Él se va a encargar de pasarlas por el Analizador o por el Demodulador FM 
        # según corresponda, y luego actualizar toda la interfaz.
        self.ui.procesar_muestras_iq(chunk)
    
    def stop_playback(self):
        self.playback_timer.stop()
        self.is_looping = False
        
        # Rehabilitar todo
        self.ui.freq_input.setEnabled(True)
        self.ui.sr_combo.setEnabled(True)
        self.ui.fft_combo.setEnabled(True)
        self.ui.record_action.setEnabled(True)
        self.ui.play_action.setEnabled(True)
        self.ui.loop_action.setEnabled(True)
        self.ui.stop_play_action.setEnabled(False)
        
        # Restaurar botón principal y color del gráfico a amarillo
        self.ui.rec_play_btn.setText("Rec/Play")
        self.ui.rec_play_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
        self.ui.freq_plot_curve.setPen(pg.mkPen(color='#FFD500', width=1.5))
        
        # Volver a encender la antena en vivo
        print("Reproducción finalizada o detenida. Volviendo a la antena.")
        if hasattr(self.ui.radio, 'start_rx'):
            self.ui.radio.start_rx()