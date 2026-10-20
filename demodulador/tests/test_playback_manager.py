"""Reproducción completa y restauración del estado de recepción."""

import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QMainWindow, QToolButton

import playback_manager as playback_module
from dsp.demoduladores.sa import SpectrumAnalyzer
from dsp.demoduladores.lte_uplink import DemoduladorLTEUplink
from dsp.stream_reset import reset_demodulator_stream
from hardware.mock_sdr_handler import MockSDRHandler
from iq_capture import IQRecorder, load_capture
from playback_manager import PlaybackManager


class Emitter(QObject):
    recording_done = pyqtSignal(object, object)


class FakeRadio:
    supports_live_rx = True

    def __init__(self):
        self.is_running = True
        self.starts = 0
        self.stops = 0

    def get_sample_rate(self):
        return 2_000_000

    def get_center_frequency(self):
        return 101_000_000

    def stop_rx(self):
        self.stops += 1
        self.is_running = False

    def start_rx(self):
        self.starts += 1
        self.is_running = True


class FakeAudio:
    def clear_audio(self):
        pass


class FakeCurve:
    def setPen(self, pen):
        pass


class FakeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.radio = FakeRadio()
        self.audio_manager = FakeAudio()
        self.freq_plot_curve = FakeCurve()
        self.current_freq_multiplier = 1_000_000
        self.freq_input = QDoubleSpinBox(self)
        self.freq_input.setRange(0, 10_000)
        self.freq_input.setValue(101)
        self.sr_combo = QComboBox(self)
        self.sr_combo.addItem("2 MHz")
        self.fft_combo = QComboBox(self)
        self.demod_btn = QToolButton(self)
        self.change_sdr_btn = QToolButton(self)
        self.btn_change_uplink_freq = QToolButton(self)
        self.rec_play_btn = QToolButton(self)
        self.record_action = QAction(self)
        self.play_action = QAction(self)
        self.loop_action = QAction(self)
        self.stop_play_action = QAction(self)
        self.demodulador_actual = SpectrumAnalyzer()
        self.demodulador_actual.configurar(2_000_000, 64)
        self.received = []

    def update_x_axis(self):
        pass

    def procesar_muestras_iq(self, samples):
        if samples is None:
            self.received.append(None)
            self.demodulador_actual = reset_demodulator_stream(self.demodulador_actual)
        else:
            self.received.append(samples.copy())


def make_capture(path, samples, sample_rate=1_000):
    recorder = IQRecorder(path, 100_000_000, sample_rate)
    for start in range(0, len(samples), 11):
        recorder.append(samples[start:start + 11])
    recorder.finish()
    assert recorder.wait()


def setup_manager(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = FakeWindow()
    state = dict(center_freq=101_000_000, sample_rate=2_000_000,
                 fft_size=64, is_recording=False)
    clock = [0.0]
    monkeypatch.setattr(playback_module, "monotonic", lambda: clock[0])
    manager = PlaybackManager(window, state, Emitter())
    return app, window, state, clock, manager


def test_replay_includes_short_last_block_and_restores_live_state(tmp_path, monkeypatch):
    samples = (np.arange(73) + 1j * np.arange(73)[::-1]).astype(np.complex64)
    path = tmp_path / "capture.iq"
    make_capture(path, samples)
    app, window, state, clock, manager = setup_manager(monkeypatch)
    monkeypatch.setattr(playback_module.QFileDialog, "getOpenFileName",
                        lambda *args: (str(path), ""))
    manager.load_and_play()
    assert window.radio.stops == 1
    assert window.change_sdr_btn.isEnabled()
    assert state['sample_rate'] == 1_000
    clock[0] = 0.2
    manager.playback_step()
    assert manager.playback_data is not None
    clock[0] = 0.31
    manager.playback_step()

    np.testing.assert_array_equal(np.concatenate(window.received), samples)
    assert manager.playback_data is None
    assert window.radio.starts == 1
    assert state['center_freq'] == 101_000_000
    assert state['sample_rate'] == 2_000_000
    assert window.sr_combo.currentText() == "2 MHz"
    window.close()


def test_offline_replay_never_starts_receiver_and_cannot_record(tmp_path, monkeypatch):
    samples = (np.arange(41) + 1j * np.arange(41)).astype(np.complex64)
    path = tmp_path / "offline.iq"
    make_capture(path, samples)
    app, window, state, clock, manager = setup_manager(monkeypatch)
    window.radio = MockSDRHandler()
    window.radio.configurar(2_000_000, 101_000_000)
    window.record_action.setEnabled(False)
    monkeypatch.setattr(playback_module.QFileDialog, "getOpenFileName",
                        lambda *args: (str(path), ""))

    manager.toggle_recording()
    assert not state['is_recording']
    assert not list(tmp_path.glob("muestras_iq_*.iq"))
    manager.load_and_play()
    clock[0] = 0.2
    manager.playback_step()
    clock[0] = 0.31
    manager.playback_step()

    np.testing.assert_array_equal(np.concatenate(window.received), samples)
    assert manager.playback_data is None
    assert not window.radio.is_running
    assert not window.record_action.isEnabled()
    window.close()


def test_loop_plays_tail_before_starting_next_cycle(tmp_path, monkeypatch):
    samples = (np.arange(37) + 1j * np.arange(37)).astype(np.complex64)
    path = tmp_path / "capture.iq"
    make_capture(path, samples)
    app, window, state, clock, manager = setup_manager(monkeypatch)
    monkeypatch.setattr(playback_module.QFileDialog, "getOpenFileName",
                        lambda *args: (str(path), ""))
    manager.load_and_play(loop=True)
    clock[0] = 0.1
    manager.playback_step()
    manager.playback_step()

    played = np.concatenate([block for block in window.received if block is not None])
    np.testing.assert_array_equal(played, np.tile(samples, 3)[:100])
    assert sum(block is None for block in window.received) == 2
    manager.stop_playback()
    assert window.radio.starts == 1
    window.close()


def test_recording_keeps_every_callback_block(tmp_path, monkeypatch):
    app, window, state, clock, manager = setup_manager(monkeypatch)
    monkeypatch.chdir(tmp_path)
    first = np.arange(7, dtype=np.float32).astype(np.complex64)
    second = (np.arange(13) * 1j).astype(np.complex64)
    manager.toggle_recording()
    assert manager.record_samples(first)
    assert manager.record_samples(second)
    manager.toggle_recording()
    manager._recorder.wait()
    app.processEvents()

    paths = list(Path(tmp_path).glob("muestras_iq_*.iq"))
    assert len(paths) == 1
    restored, center_freq, sample_rate = load_capture(paths[0])
    np.testing.assert_array_equal(restored, np.concatenate((first, second)))
    assert sample_rate == window.radio.get_sample_rate()
    assert center_freq == window.radio.get_center_frequency()
    assert manager._recorder is None
    assert window.demod_btn.isEnabled()
    window.close()


def test_reported_radio_gap_invalidates_recording(tmp_path, monkeypatch):
    app, window, state, clock, manager = setup_manager(monkeypatch)
    monkeypatch.chdir(tmp_path)
    manager.toggle_recording()
    assert manager.record_samples(np.ones(19, dtype=np.complex64))
    manager.signal_discontinuity()
    manager._recorder.wait()
    app.processEvents()

    assert not list(Path(tmp_path).glob("muestras_iq_*.iq"))
    assert not state['is_recording']
    assert "Captura descartada" in window.statusBar().currentMessage()
    window.close()


def test_stop_action_resumes_radio(tmp_path, monkeypatch):
    samples = np.arange(37, dtype=np.float32).astype(np.complex64)
    path = tmp_path / "capture.iq"
    make_capture(path, samples)
    app, window, state, clock, manager = setup_manager(monkeypatch)
    monkeypatch.setattr(playback_module.QFileDialog, "getOpenFileName",
                        lambda *args: (str(path), ""))
    manager.load_and_play(loop=True)
    window.stop_play_action.triggered.connect(lambda: manager.stop_playback())
    window.stop_play_action.trigger()
    assert window.radio.starts == 1
    assert window.radio.is_running
    assert manager.playback_data is None
    window.close()


def test_shutdown_finishes_accepted_samples_before_stopping_radio(tmp_path, monkeypatch):
    app, window, state, clock, manager = setup_manager(monkeypatch)
    monkeypatch.chdir(tmp_path)
    block = (np.arange(29) + 1j).astype(np.complex64)
    manager.toggle_recording()
    assert manager.record_samples(block)
    manager.shutdown()

    paths = list(Path(tmp_path).glob("muestras_iq_*.iq"))
    assert len(paths) == 1
    restored, _, _ = load_capture(paths[0])
    np.testing.assert_array_equal(restored, block)
    assert window.radio.stops == 1
    window.close()


def test_replay_waits_for_last_worker_result(tmp_path, monkeypatch):
    path = tmp_path / "capture.iq"
    make_capture(path, np.ones(17, dtype=np.complex64))
    app, window, state, clock, manager = setup_manager(monkeypatch)
    monkeypatch.setattr(playback_module.QFileDialog, "getOpenFileName",
                        lambda *args: (str(path), ""))
    manager.load_and_play()
    window.demodulador_actual.is_processing = True
    clock[0] = 0.2
    manager.playback_step()
    clock[0] = 0.31
    manager.playback_step()
    assert manager.playback_data is not None
    window.demodulador_actual.is_processing = False
    clock[0] = 0.32
    manager.playback_step()
    assert manager.playback_data is None
    window.close()


def test_lte_uplink_publishes_pending_result_without_filler_iq():
    demod = DemoduladorLTEUplink()
    demod.configurar(7_680_000, 512, rb_count=25)
    demod.last_heavy_results = {'metricas': {'ready': True}}
    demod.nuevos_datos_listos = True
    assert demod.procesar(np.empty(0, dtype=np.complex64)) == demod.last_heavy_results
    assert not demod.nuevos_datos_listos
