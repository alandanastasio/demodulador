"""La opción de archivos debe estar disponible aunque no se detecte una SDR."""

import numpy as np
import pytest
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication, QMainWindow

import playback_manager
import startup_window
from hardware.mock_sdr_handler import MockSDRHandler
from iq_capture import IQRecorder


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_mock_sdr_has_no_receiver():
    radio = MockSDRHandler()
    radio.configurar(2_000_000, 915_000_000)
    radio.start_rx()
    assert not radio.supports_live_rx
    assert not radio.is_running
    assert radio.get_sample_rate() == 2_000_000
    assert radio.get_center_frequency() == 915_000_000
    radio.stop_rx()
    radio.close()


def test_startup_offline_button_remains_available_without_usb(monkeypatch, qapp):
    monkeypatch.setattr(startup_window.StartupWindow, "scan_devices", lambda self: None)

    class DummyMain(QMainWindow):
        def __init__(self, radio):
            super().__init__()
            self.radio = radio

    monkeypatch.setattr(startup_window, "create_main_window", DummyMain)
    window = startup_window.StartupWindow()
    window.on_scan_completed([])
    assert window.offline_btn.isEnabled()
    assert "sin SDR" in window.offline_btn.text()
    window.offline_btn.click()
    assert isinstance(window.main_app_window.radio, MockSDRHandler)
    assert not window.main_app_window.radio.is_running
    window.main_app_window.close()


def test_main_window_replays_capture_without_receiver(tmp_path, monkeypatch, qapp):
    from main import MainWindow

    samples = (np.arange(47) + 1j * np.arange(47)).astype(np.complex64)
    path = tmp_path / "sin_sdr.iq"
    recorder = IQRecorder(path, 915_000_000, 1_000)
    recorder.append(samples)
    recorder.finish()
    assert recorder.wait()

    window = MainWindow(MockSDRHandler())
    received = []
    process = window.procesar_muestras_iq

    def track_samples(block):
        if block is not None and len(block):
            received.append(block.copy())
        return process(block)

    window.procesar_muestras_iq = track_samples
    clock = [0.0]
    monkeypatch.setattr(playback_manager, "monotonic", lambda: clock[0])
    monkeypatch.setattr(playback_manager.QFileDialog, "getOpenFileName",
                        lambda *args: (str(path), ""))

    assert not window.record_action.isEnabled()
    assert "Modo sin SDR" in window.offline_notice.text()
    assert "#493638" in window.offline_notice.styleSheet()
    window.pause_btn.click()
    assert window.is_paused and window.audio_manager._paused
    window.pause_btn.click()
    assert not window.is_paused and not window.audio_manager._paused
    window.playback_manager.load_and_play()
    assert window.playback_manager.state['sample_rate'] == 1_000
    assert window.playback_manager.playback_data is not None
    clock[0] = 0.2
    window.playback_manager.playback_step()
    clock[0] = 0.31
    window.playback_manager.playback_step()

    np.testing.assert_array_equal(np.concatenate(received), samples)
    assert window.playback_manager.playback_data is None
    assert not window.radio.is_running
    assert not window.record_action.isEnabled()
    window.close()


def test_change_sdr_button_stops_active_offline_playback(tmp_path, monkeypatch, qapp):
    from main import MainWindow

    path = tmp_path / "loop.iq"
    recorder = IQRecorder(path, 915_000_000, 1_000)
    recorder.append(np.ones(47, dtype=np.complex64))
    recorder.finish()
    assert recorder.wait()

    window = MainWindow(MockSDRHandler())
    monkeypatch.setattr(playback_manager.QFileDialog, "getOpenFileName",
                        lambda *args: (str(path), ""))
    launches = []
    monkeypatch.setattr(QProcess, "startDetached",
                        lambda program, args: launches.append((program, args)) or True)
    monkeypatch.setattr(QApplication, "quit", lambda: launches.append("quit"))

    window.playback_manager.load_and_play(loop=True)
    assert window.playback_manager.playback_timer.isActive()
    assert window.change_sdr_btn.isEnabled()
    window.change_sdr_btn.click()

    assert window.playback_manager.playback_data is None
    assert not window.playback_manager.playback_timer.isActive()
    assert len(launches) == 2
    assert launches[1] == "quit"
    window.close()


def test_both_fm_modes_show_rds_panel(qapp):
    from main import MainWindow, state
    from plot_renderer import render_plot

    window = MainWindow(MockSDRHandler())
    window.show()
    for select_mode in (window.set_wbfm_mode, window.set_wbfm_audio_mode):
        select_mode()
        qapp.processEvents()
        assert window.modes_stack.currentWidget() is window.page_wbfm
        assert window.rds_label.isVisible()
        assert window.controls_widget.layout().indexOf(window.rds_label) >= 0
        assert window.layout_wbfm.indexOf(window.rds_label) == -1
        assert 'RDS' in window.rds_label.text()
        render_plot(
            window, state, np.zeros(len(window.f_axis)),
            np.zeros(len(window.f_axis), dtype=np.complex64),
            fm_metrics={
                'pico_max': 1.0, 'pico_min': -1.0, 'pico_rms': 0.7,
                'rms': 0.5, 'dc_offset': 0.0,
                'rds': {'pi': 0xA52C, 'ps': 'RADIO123',
                        'radiotext': '<HOLA ARGENTINA>', 'pty': 10},
            },
        )
        assert 'RADIO123' in window.rds_label.text()
        assert '&lt;HOLA ARGENTINA&gt;' in window.rds_label.text()
    window.set_normal_mode()
    assert not window.rds_label.isVisible()
    window.close()
