"""Control de grabación íntegra y reproducción de muestras IQ."""

from datetime import datetime
from pathlib import Path
from threading import Lock
from time import monotonic

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QFileDialog

from dsp.demoduladores.btle import DemoduladorBTLE
from dsp.demoduladores.lora import DemoduladorLoRa
from dsp.demoduladores.lte_uplink import DemoduladorLTEUplink
from dsp.demoduladores.wbfm_audio import DemoduladorWBFMAudio
from dsp.demoduladores.wifi_ag import DemoduladorWiFiAG
from dsp.stream_reset import reset_demodulator_stream
from iq_capture import CaptureError, IQRecorder, load_capture


class PlaybackManager(QObject):
    def __init__(self, main_window, app_state, emitter):
        super().__init__(main_window)
        self.ui = main_window
        self.state = app_state
        self.emitter = emitter
        self._capture_lock = Lock()
        self._recorder = None
        self._record_controls = None
        self._play_controls = None
        self._live_state = None
        self._live_demod = None
        self.playback_data = None
        self.playback_index = 0  # Índice absoluto, incluso al reproducir en bucle.
        self.playback_started_at = None
        self.playback_finished_at = None
        self.is_looping = False
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.playback_step)
        self.emitter.recording_done.connect(self._on_recording_done)

    def _set_enabled(self, names, enabled):
        previous = {}
        for name in names:
            control = getattr(self.ui, name, None)
            if control is not None:
                previous[name] = control.isEnabled()
                control.setEnabled(enabled)
        return previous

    def _restore_enabled(self, previous):
        for name, enabled in (previous or {}).items():
            getattr(self.ui, name).setEnabled(enabled)

    def toggle_recording(self):
        if self.state['is_recording']:
            with self._capture_lock:
                self.state['is_recording'] = False
                if self._recorder is not None:
                    self._recorder.finish()
            self.ui.record_action.setEnabled(False)
            self.ui.record_action.setText("⏳ Guardando...")
            self.ui.rec_play_btn.setText("⏳ Guardando...")
            return
        if self._recorder is not None or self.playback_data is not None:
            return
        if not self.ui.radio.supports_live_rx:
            self.ui.statusBar().showMessage("No hay SDR para grabar en este modo.", 5000)
            return
        if not self.ui.radio.is_running:
            self.ui.statusBar().showMessage("El SDR debe estar recibiendo para grabar.", 5000)
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path.cwd() / f"muestras_iq_{timestamp}.iq"
        try:
            actual_rate = self.ui.radio.get_sample_rate()
            actual_frequency = self.ui.radio.get_center_frequency()
            recorder = IQRecorder(
                path,
                actual_frequency if actual_frequency is not None else self.state['center_freq'],
                actual_rate if actual_rate is not None else self.state['sample_rate'],
                on_done=lambda capture, error: self.emitter.recording_done.emit(capture, error),
            )
        except Exception as exc:
            self.ui.statusBar().showMessage(f"No se pudo iniciar la grabación: {exc}", 0)
            return

        with self._capture_lock:
            self._recorder = recorder
            self.state['is_recording'] = True
        self._record_controls = self._set_enabled(
            ('freq_input', 'sr_combo', 'fft_combo', 'demod_btn',
             'change_sdr_btn', 'btn_change_uplink_freq', 'lna_combo', 'vga_combo',
             'play_action', 'loop_action', 'wf_btn_crop'),
            False,
        )
        self.ui.record_action.setText("⏹ Detener y Guardar")
        self.ui.rec_play_btn.setText("🔴 Grabando IQ...")
        self.ui.statusBar().showMessage(
            "Grabando IQ; los gráficos se pausan para priorizar la captura.")

    def record_samples(self, samples):
        """Se llama en el callback SDR antes de cualquier procesamiento DSP."""
        with self._capture_lock:
            if not self.state['is_recording'] or self._recorder is None:
                return False
            try:
                self._recorder.append(samples)
            except CaptureError:
                self.state['is_recording'] = False
                # La señal de finalización informará la causa desde el hilo escritor.
            return True

    def signal_discontinuity(self):
        with self._capture_lock:
            if self.state['is_recording'] and self._recorder is not None:
                self.state['is_recording'] = False
                self._recorder.abort("El SDR informó un corte en el flujo IQ.")

    def _on_recording_done(self, recorder, error):
        if recorder is not self._recorder:
            return
        with self._capture_lock:
            self.state['is_recording'] = False
            self._recorder = None
        self._restore_enabled(self._record_controls)
        self._record_controls = None
        self.ui.record_action.setEnabled(self.ui.radio.supports_live_rx)
        self.ui.record_action.setText(
            "🔴 Iniciar Grabación" if self.ui.radio.supports_live_rx
            else "🔴 Grabación disponible con SDR")
        self.ui.rec_play_btn.setText("Rec/Play")
        if error:
            self.ui.statusBar().showMessage(f"Captura descartada: {error}", 0)
        else:
            count = recorder.result['sample_count']
            self.ui.statusBar().showMessage(
                f"Captura guardada: {recorder.path} ({count} muestras recibidas).", 0)

    def _make_demodulator(self, sample_rate, center_freq):
        previous = self.ui.demodulador_actual
        if previous is None:
            return None
        fresh = type(previous)()
        args = (sample_rate, self.state['fft_size'])
        if isinstance(previous, DemoduladorLoRa):
            fresh.configurar(*args, previous.bandwidth_hz, previous.sf)
        elif isinstance(previous, DemoduladorBTLE):
            fresh.configurar(*args, bw_mhz=previous.bw_mhz,
                             decode_legacy=previous.decode_legacy)
            fresh.set_center_frequency(center_freq)
        elif isinstance(previous, DemoduladorLTEUplink):
            fresh.configurar(*args, rb_count=previous.rb_count)
        elif isinstance(previous, DemoduladorWiFiAG):
            fresh.configurar(*args, remove_dc=previous.remove_dc)
        else:
            fresh.configurar(*args)
        if hasattr(previous, 'window_type'):
            fresh.set_window(previous.window_type)
        return fresh

    def load_and_play(self, loop=False):
        if self._recorder is not None or self.playback_data is not None:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self.ui, "Seleccionar grabación IQ", "",
            "Capturas IQ (*.iq *.npz)")
        if not filename:
            return
        try:
            samples, center_freq, sample_rate = load_capture(filename)
            file_demod = self._make_demodulator(sample_rate, center_freq)
        except Exception as exc:
            self.ui.statusBar().showMessage(f"No se pudo abrir la captura: {exc}", 0)
            return

        live_state = (self.state['center_freq'], self.state['sample_rate'],
                      self.ui.freq_input.value(), self.ui.sr_combo.currentText())
        self.ui._suspend_live_iq = True
        if self.ui.radio.supports_live_rx:
            try:
                self.ui.radio.stop_rx()
            except Exception as exc:
                self.ui._suspend_live_iq = False
                self.ui.statusBar().showMessage(f"No se pudo detener el SDR: {exc}", 0)
                return
        self._live_state = live_state
        self._live_demod = self.ui.demodulador_actual
        if hasattr(self.ui, '_reset_retro_iq'):
            self.ui._reset_retro_iq()
        self.ui.audio_manager.clear_audio()
        self.ui.demodulador_actual = file_demod
        if hasattr(self.ui, '_reset_rds_display'):
            self.ui._reset_rds_display()
        if isinstance(file_demod, DemoduladorWBFMAudio):
            self.ui._bind_wbfm_audio_result(file_demod)
        self.playback_data = samples
        self.playback_index = 0
        self.playback_finished_at = None
        self.is_looping = bool(loop)
        self.state['center_freq'] = center_freq
        self.state['sample_rate'] = sample_rate
        self._play_controls = self._set_enabled(
            ('freq_input', 'sr_combo', 'fft_combo', 'demod_btn',
             'btn_change_uplink_freq', 'lna_combo', 'vga_combo',
             'record_action', 'play_action', 'loop_action'), False)
        self.ui.stop_play_action.setEnabled(True)
        self.ui.freq_input.blockSignals(True)
        self.ui.freq_input.setValue(center_freq / self.ui.current_freq_multiplier)
        self.ui.freq_input.blockSignals(False)
        rate_text = f"{sample_rate / 1e6:g} MHz"
        self._added_rate_text = rate_text if self.ui.sr_combo.findText(rate_text) < 0 else None
        if self._added_rate_text:
            self.ui.sr_combo.addItem(rate_text)
        self.ui.sr_combo.blockSignals(True)
        self.ui.sr_combo.setCurrentText(rate_text)
        self.ui.sr_combo.blockSignals(False)
        self.ui.update_x_axis()
        self.ui.freq_plot_curve.setPen(pg.mkPen(color="#FF8C00", width=1.5))
        self.ui.rec_play_btn.setText("🔁 Reproduciendo..." if loop else "▶ Reproduciendo...")
        self.playback_started_at = monotonic()
        self.playback_timer.start(10)

    def playback_step(self):
        if self.playback_data is None:
            return
        count = len(self.playback_data)
        target = int((monotonic() - self.playback_started_at) * self.state['sample_rate'])
        if not self.is_looping:
            target = min(target, count)
        max_chunk = max(1, int(self.state['sample_rate'] * 0.025))
        # Recuperar retrasos del event loop sin descartar ni repetir muestras.
        for _ in range(4):
            if self.playback_index >= target:
                break
            start = self.playback_index % count
            if self.is_looping and start == 0 and self.playback_index:
                self.ui.procesar_muestras_iq(None)
            size = min(target - self.playback_index, count - start, max_chunk)
            block = np.array(self.playback_data[start:start + size], copy=True)
            self.playback_index += size
            self.ui.procesar_muestras_iq(block)
        if not self.is_looping and self.playback_index == count:
            now = monotonic()
            if self.playback_finished_at is None:
                self.playback_finished_at = now
            # Algunos demoduladores publican el último resultado desde un
            # worker o en la siguiente llamada. Consultarlos sin añadir IQ.
            self.ui.procesar_muestras_iq(np.empty(0, dtype=self.playback_data.dtype))
            elapsed = now - self.playback_finished_at
            if elapsed >= 10:
                self.ui.statusBar().showMessage(
                    "Se entregó todo el IQ, pero el procesador tardó en finalizar.", 0)
                self.stop_playback()
            elif elapsed >= 0.1 and not self._demod_pending():
                self.stop_playback()

    def _demod_pending(self):
        demod = self.ui.demodulador_actual
        if demod is None:
            return False
        for name in ('is_processing', '_processing', '_worker_active',
                     'nuevos_datos_listos'):
            if getattr(demod, name, False):
                return True
        for name in ('_pending_blocks', '_completed_results', '_ready_results',
                     '_pending_frames'):
            if len(getattr(demod, name, ())):
                return True
        return False

    def stop_playback(self, restart_radio=True):
        if self.playback_data is None:
            return
        self.playback_timer.stop()
        self.ui.audio_manager.clear_audio()
        if isinstance(self.ui.demodulador_actual, DemoduladorLoRa):
            self.ui.demodulador_actual.close()
        self.playback_data = None
        self.playback_index = 0
        self.playback_started_at = None
        self.playback_finished_at = None
        self.is_looping = False
        center_freq, sample_rate, freq_value, rate_text = self._live_state
        self.state['center_freq'] = center_freq
        self.state['sample_rate'] = sample_rate
        if hasattr(self.ui, '_reset_retro_iq'):
            self.ui._reset_retro_iq()
        self.ui.demodulador_actual = reset_demodulator_stream(self._live_demod)
        if hasattr(self.ui, '_reset_rds_display'):
            self.ui._reset_rds_display()
        if isinstance(self.ui.demodulador_actual, DemoduladorWBFMAudio):
            self.ui._bind_wbfm_audio_result(self.ui.demodulador_actual)
        self._live_demod = None
        self._live_state = None
        self.ui.freq_input.blockSignals(True)
        self.ui.freq_input.setValue(freq_value)
        self.ui.freq_input.blockSignals(False)
        self.ui.sr_combo.blockSignals(True)
        self.ui.sr_combo.setCurrentText(rate_text)
        if self._added_rate_text:
            index = self.ui.sr_combo.findText(self._added_rate_text)
            if index >= 0:
                self.ui.sr_combo.removeItem(index)
        self.ui.sr_combo.blockSignals(False)
        self._added_rate_text = None
        self.ui.update_x_axis()
        self._restore_enabled(self._play_controls)
        self._play_controls = None
        self.ui.stop_play_action.setEnabled(False)
        self.ui.rec_play_btn.setText("Rec/Play")
        self.ui.freq_plot_curve.setPen(pg.mkPen(color="#FFD500", width=1.5))
        if restart_radio and self.ui.radio.supports_live_rx:
            self.ui._suspend_live_iq = False
            self.ui.radio.start_rx()

    def shutdown(self):
        if self.playback_data is not None:
            self.stop_playback(restart_radio=False)
        with self._capture_lock:
            self.state['is_recording'] = False
            recorder = self._recorder
            if recorder is not None:
                recorder.finish()
        if self.ui.radio.supports_live_rx:
            self.ui.radio.stop_rx()
        if recorder is not None:
            recorder.wait()
