"""Regresiones del receptor OFDM 802.11a/g sin depender de una SDR."""

import threading
import time
import warnings

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


def test_lts_fine_sync_finds_preamble_with_offset_cfo_and_noise():
    rng = np.random.default_rng(17)
    short = (rng.choice([-1, 1], 16) + 1j * rng.choice([-1, 1], 16)) * 0.07
    preamble = np.r_[np.tile(short, 10), wifi_ag.LTS_TIME[-32:],
                     wifi_ag.LTS_TIME, wifi_ag.LTS_TIME]
    for offset, cfo_hz in [(0, 0), (37, 40_000), (71, -85_000)]:
        signal = np.r_[np.zeros(offset), preamble, np.zeros(120)]
        signal = signal * np.exp(2j * np.pi * cfo_hz * np.arange(len(signal)) / 20e6)
        signal += (rng.normal(size=len(signal)) + 1j * rng.normal(size=len(signal))) * 0.002
        metric, corr, energy = wifi_ag.schmidl_cox_metric(signal)
        indices = np.flatnonzero((energy > 0.1 * np.max(energy)) & (metric > 0.7))
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        plateau = max(runs, key=len)
        estimate = np.angle(np.mean(corr[plateau])) * 20e6 / (2 * np.pi * 16)
        lts = wifi_ag.find_lts_start(signal, estimate, 20e6,
                                     int(plateau[0]), int(plateau[-1]))
        assert lts == offset + 192


def test_lts_sync_rejects_repeated_tone_without_training_sequence():
    signal = np.ones(800, dtype=np.complex64)
    assert wifi_ag.find_lts_start(signal, 0, 20e6, 0, 127) is None


def test_evm_uses_nominal_constellation_power_and_excludes_invalid_bins():
    data_bins = [38, 39, 1, 2]
    pilot_bins = [7, 21, 43, 57]
    data = np.full((2, 4), 1.1 + 0j)
    pilots = np.ones((2, 4), dtype=complex)
    valid_data = np.array([True, True, True, False])
    valid_pilots = np.array([True, False, True, True])
    data[:, -1] = 100  # Este nulo de canal no debe contaminar el resultado.
    pilots[:, 1] = 100
    evm = wifi_ag.calculate_evm(data, pilots, np.ones_like(pilots), 'BPSK',
                                data_bins, pilot_bins, valid_data, valid_pilots)
    assert len(evm['subc_x']) == 6
    np.testing.assert_allclose(evm['sym_rms'],
                               20 * np.log10(0.1 * np.sqrt(3 / 6)), atol=1e-10)
    np.testing.assert_allclose(evm['subc_rms'][evm['subc_x'] == 1], -20, atol=1e-10)


def test_evm_recognizes_unit_power_16qam_constellation():
    data = np.array([[(3 + 1j) / np.sqrt(10), (-1 - 3j) / np.sqrt(10)]])
    pilots = np.array([[1, -1]], dtype=complex)
    evm = wifi_ag.calculate_evm(data, pilots, pilots, '16-QAM',
                                [38, 1], [7, 43],
                                np.array([True, True]), np.array([True, True]))
    assert np.max(evm['sym_rms']) < -190


def test_complete_synthetic_6mbps_packet_decodes_with_cfo():
    rng = np.random.default_rng(23)
    short = (rng.choice([-1, 1], 16) + 1j * rng.choice([-1, 1], 16)) * 0.07
    preamble = np.r_[np.tile(short, 10), wifi_ag.LTS_TIME[-32:],
                     wifi_ag.LTS_TIME, wifi_ag.LTS_TIME]
    data_bins = [i for i in list(range(38, 64)) + list(range(1, 27))
                 if i not in (43, 57, 7, 21)]
    pilot_bins = [7, 21, 43, 57]
    pilot_ref = np.array([1, -1, 1, 1])
    pn = wifi_ag.pilot_polarities(3)

    # SIGNAL: RATE=1101 (6 Mbit/s), LENGTH=1, paridad par y tail=0.
    info = np.zeros(24, dtype=np.int8)
    info[:4] = [1, 1, 0, 1]
    info[5] = 1
    info[17] = np.sum(info[:17]) % 2
    coded = np.empty(48, dtype=np.int8)
    state = 0
    for n, bit in enumerate(info):
        reg = (int(bit) << 6) | state
        coded[2 * n] = (reg & 0b1011011).bit_count() % 2
        coded[2 * n + 1] = (reg & 0b1111001).bit_count() % 2
        state = (int(bit) << 5) | (state >> 1)
    interleaved = np.empty(48, dtype=np.int8)
    for k in range(48):
        interleaved[3 * (k % 16) + k // 16] = coded[k]

    def ofdm_symbol(data, symbol_no):
        freq = np.zeros(64, dtype=complex)
        freq[data_bins] = data
        freq[pilot_bins] = pilot_ref * pn[symbol_no]
        samples = np.fft.ifft(freq)
        return np.r_[samples[-16:], samples]

    signal_symbol = ofdm_symbol(2 * interleaved - 1, 0)
    data_symbols = [ofdm_symbol(rng.choice([-1, 1], 48), n) for n in (1, 2)]
    packet = np.r_[preamble, signal_symbol, *data_symbols]
    iq = np.r_[np.zeros(400), packet, np.zeros(120)]
    iq *= np.exp(2j * np.pi * 40_000 * np.arange(len(iq)) / 20e6)
    iq += (rng.normal(size=len(iq)) + 1j * rng.normal(size=len(iq))) * 0.0003

    demod = wifi_ag.DemoduladorWiFiAG()
    demod.configurar(20e6, 4096)
    demod._procesar_fondo(iq.astype(np.complex64))

    metrics = demod.ultimo_wifi_metrics
    assert metrics['mbps'] == 6
    assert metrics['length'] == 1
    assert metrics['paridad_ok'] and metrics['tail_ok']
    assert len(demod.ultimo_evm_data['sym_rms']) == 2
    assert max(demod.ultimo_evm_data['sym_rms']) < -25


def test_channel_equalizer_rejects_invalid_lts_and_handles_a_notched_tone():
    active = np.r_[1:27, 38:64]
    channel = np.zeros(64, dtype=complex)
    assert wifi_ag.channel_equalizer(channel, active) == (None, None)

    channel[active] = 2 + 1j
    channel[7] = 0
    weights, floor = wifi_ag.channel_equalizer(channel, active)
    assert floor > 0
    assert np.all(np.isfinite(weights))
    assert weights[7] == 0
    np.testing.assert_allclose(channel[8] * weights[8], 1, rtol=2e-6)


def test_false_preamble_with_zero_lts_does_not_divide_by_zero():
    demod = wifi_ag.DemoduladorWiFiAG()
    demod.configurar(20e6, 4096)
    iq = np.zeros(65_536, dtype=np.complex64)
    iq[500:6500] = 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        demod._procesar_fondo(iq)

    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert demod.ultimo_wifi_metrics == {}
    assert len(demod.last_heavy_results['psd_rf']) == 4096


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
