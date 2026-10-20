"""Continuidad y formato de las capturas IQ."""

import json

import numpy as np
import pytest

from iq_capture import CaptureError, IQRecorder, load_capture


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_irregular_blocks_round_trip_without_phase_or_sample_loss(tmp_path, dtype):
    sample_rate = 4_000
    count = 1_003
    offset_hz = 173
    source = np.exp(2j * np.pi * offset_hz * np.arange(count) / sample_rate).astype(dtype)
    path = tmp_path / "capture.iq"
    recorder = IQRecorder(path, 917_500_000, sample_rate)
    cuts = (1, 27, 109, 110, 432, 999, count)
    previous = 0
    for end in cuts:
        recorder.append(source[previous:end])
        previous = end
    recorder.finish()
    metadata = recorder.wait()

    restored, center_freq, rate = load_capture(path)
    assert metadata["sample_count"] == count
    assert metadata["block_count"] == len(cuts)
    assert json.loads(path.with_suffix(".json").read_text())["dtype"] == source.dtype.str
    assert center_freq == 917_500_000
    assert rate == sample_rate
    np.testing.assert_array_equal(restored, source)
    phase_step = np.angle(restored[1:] * np.conj(restored[:-1]))
    assert np.max(np.abs(phase_step - 2 * np.pi * offset_hz / sample_rate)) < 1e-5


def test_truncated_or_modified_capture_is_rejected(tmp_path):
    path = tmp_path / "capture.iq"
    recorder = IQRecorder(path, 100_000_000, 2_000_000)
    recorder.append(np.arange(23, dtype=np.float32).astype(np.complex64))
    recorder.finish()
    assert recorder.wait()

    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(CaptureError, match="incompleta"):
        load_capture(path)
    path.write_bytes(original[:-16] + bytes(16))
    with pytest.raises(CaptureError, match="integridad"):
        load_capture(path)


def test_queue_overflow_discards_partial_capture(tmp_path):
    path = tmp_path / "overflow.iq"
    recorder = IQRecorder(path, 100_000_000, 2_000_000, max_pending_bytes=8)
    with pytest.raises(CaptureError, match="cola IQ llena"):
        recorder.append(np.ones(2, dtype=np.complex64))
    assert recorder.wait() is None
    assert not path.exists()
    assert not path.with_suffix(".json").exists()


def test_reported_discontinuity_discards_capture(tmp_path):
    path = tmp_path / "gap.iq"
    recorder = IQRecorder(path, 100_000_000, 2_000_000)
    recorder.append(np.ones(17, dtype=np.complex64))
    recorder.abort("Corte del SDR")
    assert recorder.wait() is None
    assert recorder.error == "Corte del SDR"
    assert not path.exists()


def test_old_npz_capture_remains_readable(tmp_path):
    path = tmp_path / "old.npz"
    samples = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    np.savez(path, raw_iq=samples, center_freq=100_000_000, sample_rate=2_000_000)
    restored, center_freq, sample_rate = load_capture(path)
    np.testing.assert_array_equal(restored, samples)
    assert center_freq == 100_000_000
    assert sample_rate == 2_000_000
