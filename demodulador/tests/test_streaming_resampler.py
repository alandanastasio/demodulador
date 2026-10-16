"""El remuestreo por bloques equivale a procesar la captura completa."""

import numpy as np
import pytest
from scipy.signal import resample_poly

from dsp.streaming_resampler import StreamingResampler


@pytest.mark.parametrize('up, down', [(1, 10), (4, 25), (3, 7)])
def test_streaming_matches_whole_signal(up, down):
    rng = np.random.default_rng(7)
    signal = rng.normal(size=(100_003, 2))
    resampler = StreamingResampler(up, down)
    limits = [0, 500, 3401, 21_043, 40_000, len(signal)]
    pieces = [resampler.process(signal[a:b])
              for a, b in zip(limits[:-1], limits[1:])]
    output = np.concatenate(pieces)
    expected = resample_poly(signal, up, down, axis=0, window=('kaiser', 8.6))

    np.testing.assert_allclose(output, expected[:len(output)], atol=1e-12)
    assert len(output) >= len(expected) - 20
