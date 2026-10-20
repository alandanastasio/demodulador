"""Las transiciones RTL-SDR esperan a que termine la lectura USB."""

from threading import Event

import pytest

from hardware import rtlsdr_handler
from hardware.stream_transition import receiver_transition


class FakeRtlSdr:
    def __init__(self):
        self.gain = None
        self.sample_rate = 2_400_000
        self._center_freq = 100_000_000
        self.reading = Event()
        self.cancelled = Event()
        self.starts = 0
        self.stops = 0
        self.closed = False

    @property
    def center_freq(self):
        return self._center_freq

    @center_freq.setter
    def center_freq(self, value):
        assert not self.reading.is_set()
        self._center_freq = value

    def read_samples_async(self, callback, block_size):
        self.cancelled.clear()
        self.starts += 1
        self.reading.set()
        assert self.cancelled.wait(2)
        self.reading.clear()

    def cancel_read_async(self):
        self.stops += 1
        self.cancelled.set()

    def close(self):
        self.closed = True


def test_retune_pauses_rtl_stream_before_accessing_tuner(monkeypatch):
    device = FakeRtlSdr()
    monkeypatch.setattr(rtlsdr_handler, 'RtlSdr', lambda: device)
    handler = rtlsdr_handler.RtlSdrHandler(lambda _: None)
    handler.start_rx()
    assert device.reading.wait(2)

    class Host:
        radio = handler

        @receiver_transition
        def retune(self):
            self.radio.set_freq(100_700_000)

    try:
        Host().retune()
        assert device.center_freq == 100_700_000
        assert handler.is_running
        assert device.reading.wait(2)
        assert device.starts == 2
        assert device.stops == 1
    finally:
        handler.close()
    assert device.closed


def test_failed_retune_does_not_restart_rtl_stream(monkeypatch):
    class FailingRtlSdr(FakeRtlSdr):
        @FakeRtlSdr.center_freq.setter
        def center_freq(self, value):
            assert not self.reading.is_set()
            raise OSError('falló el sintonizador')

    device = FailingRtlSdr()
    monkeypatch.setattr(rtlsdr_handler, 'RtlSdr', lambda: device)
    handler = rtlsdr_handler.RtlSdrHandler(lambda _: None)
    handler.start_rx()
    assert device.reading.wait(2)

    class Host:
        radio = handler

        @receiver_transition
        def retune(self):
            self.radio.set_freq(100_700_000)

    with pytest.raises(OSError, match='falló el sintonizador'):
        Host().retune()
    assert not handler.is_running
    assert device.starts == 1
    assert device.stops == 1
    handler.close()
