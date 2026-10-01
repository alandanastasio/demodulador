"""Regresiones del receptor OFDM 802.11a/g sin depender de una SDR."""

import threading
import time

import numpy as np

from dsp.demoduladores import wifi_ag


def wait_for_idle(demod, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with demod._lock:
            if not demod.is_processing and not demod._pending_blocks:
                return
        time.sleep(0.005)
    raise AssertionError("El worker WiFi no terminó")


def test_pilot_polarities_match_80211_ag_and_repeat_after_127_symbols():
    polarities = wifi_ag.pilot_polarities(128)
    np.testing.assert_array_equal(
        polarities[:16],
        [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1],
    )
    assert polarities[127] == polarities[0]
    assert polarities[1] == 1  # Primer símbolo DATA, después de SIGNAL.


def test_single_first_burst_is_sent_to_preamble_detector(monkeypatch):
    demod = wifi_ag.DemoduladorWiFiAG()
    demod.configurar(20e6, 4096)
    iq = np.zeros(4096, dtype=np.complex64)
    iq[500:2500] = 1
    calls = []
    original = wifi_ag.schmidl_cox_metric

    def record(signal, *args, **kwargs):
        calls.append(len(signal))
        return original(signal, *args, **kwargs)

    monkeypatch.setattr(wifi_ag, 'schmidl_cox_metric', record)
    demod._procesar_fondo(iq)

    assert len(calls) == 1
    assert len(demod.last_heavy_results['psd_rf']) == 4096


def test_burst_crossing_callback_boundary_is_processed_complete(monkeypatch):
    demod = wifi_ag.DemoduladorWiFiAG()
    demod.configurar(20e6, 4096)
    first = np.r_[np.zeros(200), np.ones(800)].astype(np.complex64)
    second = np.r_[np.ones(800), np.zeros(400)].astype(np.complex64)
    calls = []
    original = wifi_ag.schmidl_cox_metric

    def record(signal, *args, **kwargs):
        calls.append(len(signal))
        return original(signal, *args, **kwargs)

    monkeypatch.setattr(wifi_ag, 'schmidl_cox_metric', record)
    demod.procesar(first)
    demod.procesar(second)
    wait_for_idle(demod)

    assert len(calls) == 1
    assert calls[0] > 1600
    assert demod.dropped_blocks == 0


def test_incoming_blocks_are_queued_while_worker_is_busy(monkeypatch):
    demod = wifi_ag.DemoduladorWiFiAG()
    demod.configurar(20e6, 4096)
    gate = threading.Event()
    started = threading.Event()
    seen = []

    def process(block, min_new_end, generation):
        seen.append((block.copy(), min_new_end))
        started.set()
        if len(seen) == 1:
            assert gate.wait(2)

    monkeypatch.setattr(demod, '_procesar_fondo', process)
    try:
        demod.procesar(np.full(4, 1, dtype=np.complex64))
        assert started.wait(1)
        demod.procesar(np.full(4, 2, dtype=np.complex64))
        demod.procesar(np.full(4, 3, dtype=np.complex64))
    finally:
        gate.set()
    wait_for_idle(demod)

    assert len(seen) == 3
    np.testing.assert_array_equal(seen[1][0], [1, 1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(seen[2][0], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    assert [entry[1] for entry in seen] == [0, 4, 8]
    assert demod.dropped_blocks == 0


def test_reconfiguration_discards_result_from_previous_worker(monkeypatch):
    demod = wifi_ag.DemoduladorWiFiAG()
    demod.configurar(20e6, 4096)
    iq = np.zeros(4096, dtype=np.complex64)
    iq[500:2500] = 1
    entered = threading.Event()
    resume = threading.Event()
    original = wifi_ag.schmidl_cox_metric

    def wait_in_detector(signal, *args, **kwargs):
        entered.set()
        assert resume.wait(2)
        return original(signal, *args, **kwargs)

    monkeypatch.setattr(wifi_ag, 'schmidl_cox_metric', wait_in_detector)
    try:
        demod.procesar(iq)
        assert entered.wait(1)
        demod.configurar(20e6, 8192)
    finally:
        resume.set()
    wait_for_idle(demod)

    assert not demod.nuevos_datos_listos
    assert demod.last_heavy_results == {}
    assert demod.ultimo_wifi_metrics == {}


def test_receiving_a_display_result_does_not_discard_new_iq(monkeypatch):
    demod = wifi_ag.DemoduladorWiFiAG()
    demod.configurar(20e6, 4096)
    seen = []

    def process(block, min_new_end, generation):
        seen.append(block.copy())
        if len(seen) == 1:
            with demod._lock:
                demod._completed_results.append({'first': True})
                demod.nuevos_datos_listos = True

    monkeypatch.setattr(demod, '_procesar_fondo', process)
    demod.procesar(np.full(4, 1, dtype=np.complex64))
    wait_for_idle(demod)
    result = demod.procesar(np.full(4, 2, dtype=np.complex64))
    wait_for_idle(demod)

    assert result == {'first': True}
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[1], [1, 1, 1, 1, 2, 2, 2, 2])
