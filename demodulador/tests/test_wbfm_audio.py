"""Separación estéreo de FM en la ruta de audio por bloques."""

import numpy as np
import pytest

from dsp.demoduladores.wbfm_audio import DemoduladorWBFMAudio


@pytest.mark.parametrize('pilot_phase', [0.0, 0.41])
def test_live_audio_separates_left_and_right_across_blocks(pilot_phase):
    sample_rate = 3_000_000
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
        result = demod.procesar(iq[start:start + block_size])

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
