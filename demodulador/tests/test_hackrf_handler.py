"""Contrato del handler sin conectar hardware USB."""

from types import SimpleNamespace

import numpy as np
import pytest

from hardware import hackrf_handler
from hardware.stream_transition import receiver_transition


class FakeHackRF:
    def __init__(self):
        self.calls = []
        self.streaming = False
        self.fail_rate = None
        self.fail_start = False

    def set_rx_callback(self, callback):
        self.callback = callback

    def pyhackrf_is_streaming(self):
        return self.streaming

    def pyhackrf_set_sample_rate(self, rate):
        self.calls.append(('rate', rate))
        if rate == self.fail_rate:
            self.fail_rate = None
            raise RuntimeError('USB rate error')

    def pyhackrf_set_baseband_filter_bandwidth(self, bandwidth):
        self.calls.append(('filter', bandwidth))

    def pyhackrf_set_freq(self, freq):
        self.calls.append(('freq', freq))

    def pyhackrf_set_lna_gain(self, gain):
        self.calls.append(('lna', gain))

    def pyhackrf_set_vga_gain(self, gain):
        self.calls.append(('vga', gain))

    def pyhackrf_start_rx(self):
        self.calls.append(('start', None))
        if self.fail_start:
            raise RuntimeError('USB start error')
        self.streaming = True

    def pyhackrf_stop_rx(self):
        self.calls.append(('stop', None))
        self.streaming = False

    def pyhackrf_close(self):
        self.calls.append(('close', None))


@pytest.fixture
def fake_library(monkeypatch):
    device = FakeHackRF()
    counts = {'init': 0, 'exit': 0}

    def init():
        counts['init'] += 1

    def exit_library():
        counts['exit'] += 1

    library = SimpleNamespace(
        pyhackrf_init=init,
        pyhackrf_exit=exit_library,
        pyhackrf_open=lambda: device,
        pyhackrf_compute_baseband_filter_bw_round_down_lt=lambda value: value,
    )
    monkeypatch.setattr(hackrf_handler, 'pyhackrf', library)
    return device, counts, library


def test_invalid_rate_does_not_interrupt_stream(fake_library):
    device, _, _ = fake_library
    handler = hackrf_handler.HackRFHandler(lambda _: None)
    handler.configurar(10_000_000, 100_000_000)
    handler.start_rx()
    before = list(device.calls)

    for rate in (1_920_000, 23_040_000, 30_720_000):
        with pytest.raises(ValueError, match='2 a 20 Msps'):
            handler.set_sample_rate(rate)
    with pytest.raises(ValueError, match='1 MHz y 6 GHz'):
        handler.set_freq(0)

    assert handler.is_running
    assert device.calls == before
    handler.close()


def test_retune_and_rate_change_mark_gap_before_new_samples(fake_library):
    device, _, _ = fake_library
    received = []
    handler = hackrf_handler.HackRFHandler(received.append)
    handler.configurar(10_000_000, 100_000_000)
    handler.start_rx()
    handler.set_freq(101_000_000)
    handler.set_sample_rate(8_000_000)

    assert received == [None, None]
    assert device.calls[-7:] == [
        ('stop', None), ('freq', 101_000_000), ('start', None),
        ('stop', None), ('rate', 8_000_000), ('filter', 6_000_000), ('start', None),
    ]
    assert handler.is_running
    handler.close()


def test_wifi_can_use_20_mhz_filter_and_other_modes_restore_default(fake_library):
    device, _, _ = fake_library
    handler = hackrf_handler.HackRFHandler(lambda _: None)
    handler.configurar(20_000_000, 2_412_000_000)
    handler.start_rx()

    assert handler._baseband_bandwidth == 15_000_000
    handler.set_baseband_filter_bandwidth(20_000_000)
    assert handler._baseband_bandwidth == 20_000_000
    assert handler.is_running
    handler.set_baseband_filter_bandwidth(None)
    assert handler._baseband_bandwidth == 15_000_000
    assert device.calls[-2:] == [('filter', 15_000_000), ('start', None)]

    with pytest.raises(ValueError, match='no puede superar'):
        handler.set_baseband_filter_bandwidth(24_000_000)
    handler.close()


def test_failed_rate_restores_previous_running_configuration(fake_library):
    device, _, _ = fake_library
    received = []
    handler = hackrf_handler.HackRFHandler(received.append)
    handler.configurar(10_000_000, 100_000_000)
    handler.start_rx()
    device.fail_rate = 8_000_000

    with pytest.raises(RuntimeError, match='USB rate error'):
        handler.set_sample_rate(8_000_000)

    assert handler.is_running
    assert handler._sample_rate == 10_000_000
    assert device.calls[-3:] == [('rate', 10_000_000), ('filter', 7_500_000), ('start', None)]
    assert received == [None]
    handler.close()


def test_start_failure_and_close_are_consistent_and_idempotent(fake_library):
    device, counts, _ = fake_library
    handler = hackrf_handler.HackRFHandler(lambda _: None)
    handler.configurar(10_000_000, 100_000_000)
    device.fail_start = True

    with pytest.raises(RuntimeError, match='USB start error'):
        handler.start_rx()
    assert not handler.is_running

    handler.close()
    handler.close()
    assert device.calls.count(('close', None)) == 1
    assert counts == {'init': 1, 'exit': 1}


def test_open_failure_exits_library(fake_library):
    _, counts, library = fake_library
    library.pyhackrf_open = lambda: None
    with pytest.raises(RuntimeError, match='abrir'):
        hackrf_handler.HackRFHandler(None)
    assert counts == {'init': 1, 'exit': 1}


def test_iq_callback_keeps_interleaved_signed_samples(fake_library):
    device, _, _ = fake_library
    received = []
    handler = hackrf_handler.HackRFHandler(received.append)
    handler.configurar(2_000_000, 100_000_000)
    handler.start_rx()
    raw = np.array([127, -128, -64, 32], dtype=np.int8)

    assert device.callback(device, raw, len(raw), len(raw)) == 0
    np.testing.assert_allclose(received[-1], [127 / 128 - 1j, -0.5 + 0.25j])
    handler.close()


def test_nested_mode_change_pauses_once_until_configuration_is_complete(fake_library):
    device, _, _ = fake_library
    received = []
    handler = hackrf_handler.HackRFHandler(received.append)
    handler.configurar(10_000_000, 100_000_000)
    handler.start_rx()

    class Host:
        radio = handler

        @receiver_transition
        def change_mode(self):
            assert not self.radio.is_running
            self.radio.set_sample_rate(8_000_000)
            self.retune()
            assert not self.radio.is_running

        @receiver_transition
        def retune(self):
            self.radio.set_freq(101_000_000)

    Host().change_mode()

    assert handler.is_running
    assert received == [None, None]
    assert [call[0] for call in device.calls].count('stop') == 1
    assert [call[0] for call in device.calls].count('start') == 2
    handler.close()


def test_qaction_triggered_can_switch_to_no_arg_modes(fake_library, monkeypatch):
    # Reproduce el clic del menú: QAction.triggered(bool) llega al decorador.
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    device, _, _ = fake_library
    handler = hackrf_handler.HackRFHandler(lambda _: None)
    handler.configurar(2_000_000, 917_500_000)
    handler.start_rx()

    class Host:
        radio = handler

        def __init__(self):
            self.modes = []

        @receiver_transition
        def set_normal_mode(self):
            self.modes.append('normal')

        @receiver_transition
        def set_wbfm_mode(self):
            self.modes.append('wbfm')

    host = Host()
    for name, callback in [('normal', host.set_normal_mode), ('wbfm', host.set_wbfm_mode)]:
        action = QAction(name)
        action.setCheckable(True)
        action.triggered.connect(callback)
        action.trigger()

    assert host.modes == ['normal', 'wbfm']
    assert handler.is_running
    assert [call[0] for call in device.calls].count('stop') == 2
    handler.close()
