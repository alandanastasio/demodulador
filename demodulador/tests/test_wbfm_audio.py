"""Separación estéreo de FM en la ruta de audio por bloques."""

import numpy as np
import pytest
from time import monotonic, sleep
from threading import Event

from dsp.demoduladores.wbfm_audio import DemoduladorWBFMAudio


def process_block(demod, iq):
    demod.procesar(iq)
    deadline = monotonic() + 3
    result = None
    while result is None and monotonic() < deadline:
        result = demod.procesar(np.empty(0, dtype=np.complex64))
        if result is None:
            sleep(0.005)
    assert result is not None
    return result


@pytest.mark.parametrize('pilot_phase', [0.0, 0.41])
@pytest.mark.parametrize('sample_rate', [2_400_000, 3_000_000])
def test_live_audio_separates_left_and_right_across_blocks(pilot_phase, sample_rate):
    block_size = sample_rate // 10
    time_s = np.arange(3 * block_size) / sample_rate

    left = 12_000 * np.cos(2 * np.pi * 1_000 * time_s)
    right = 12_000 * np.cos(2 * np.pi * 3_000 * time_s)
    pilot = 2 * np.pi * 19_000 * time_s + pilot_phase
    # La subportadora de diferencia sigue la fase del piloto FM estéreo.
    multiplex = ((left + right) - (left - right) * np.sin(2 * pilot)
                 + 5_000 * np.cos(pilot))
    iq = np.exp(2j * np.pi * np.cumsum(multiplex) / sample_rate).astype(np.complex64)

    demod = DemoduladorWBFMAudio()
    demod.configurar(sample_rate, 4096)
    for start in range(0, len(iq), block_size):
        result = process_block(demod, iq[start:start + block_size])

    audio = result['audio_out'][960:]
    assert audio.shape == (3840, 2)
    audio_time = np.arange(len(audio)) / 48_000

    def tone_amplitude(channel, frequency):
        carrier = np.exp(-2j * np.pi * frequency * audio_time)
        return abs(2 * np.mean(audio[:, channel] * carrier))

    left_tone = tone_amplitude(0, 1_000)
    right_tone = tone_amplitude(1, 3_000)
    left_leakage = tone_amplitude(1, 1_000)
    right_leakage = tone_amplitude(0, 3_000)

    assert 20 * np.log10(left_tone / left_leakage) > 25
    assert 20 * np.log10(right_tone / right_leakage) > 25
    assert result['metricas']['stereo_locked']


def test_live_audio_without_pilot_plays_identical_mono_channels():
    sample_rate = 3_000_000
    time_s = np.arange(sample_rate // 10) / sample_rate
    multiplex = 12_000 * np.cos(2 * np.pi * 1_000 * time_s)
    iq = np.exp(2j * np.pi * np.cumsum(multiplex) / sample_rate).astype(np.complex64)

    demod = DemoduladorWBFMAudio()
    demod.configurar(sample_rate, 4096)
    result = process_block(demod, iq)

    assert not result['metricas']['stereo_locked']
    np.testing.assert_allclose(result['audio_out'][:, 0],
                               result['audio_out'][:, 1], atol=1e-6)


def test_backlogged_worker_keeps_latest_audio_block(monkeypatch):
    demod = DemoduladorWBFMAudio()
    demod.configurar(3_000_000, 4096)
    started = Event()
    release = Event()

    def slow_block(iq):
        if iq[0].real == 1:
            started.set()
            assert release.wait(2)
        return {'marker': iq[0].real}

    monkeypatch.setattr(demod, '_procesar_bloque', slow_block)
    try:
        demod.procesar(np.full(300_000, 1, dtype=np.complex64))
        assert started.wait(2)
        for marker in (2, 3, 4):
            demod.procesar(np.full(300_000, marker, dtype=np.complex64))
    finally:
        release.set()

    deadline = monotonic() + 3
    result = None
    while result is None and monotonic() < deadline:
        result = demod.procesar(np.empty(0, dtype=np.complex64))
        if result is None:
            sleep(0.005)
    assert result == {'marker': 4}


def test_worker_delivers_audio_without_waiting_for_next_iq_callback(monkeypatch):
    demod = DemoduladorWBFMAudio()
    demod.configurar(3_000_000, 4096)
    delivered = Event()
    results = []
    monkeypatch.setattr(demod, '_procesar_bloque',
                        lambda iq: {'marker': iq[0].real})
    demod.result_callback = lambda result: (results.append(result), delivered.set())

    assert demod.procesar(np.full(300_000, 7, dtype=np.complex64)) is None
    assert delivered.wait(2)
    assert results == [{'marker': 7}]
