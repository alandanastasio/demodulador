"""El adaptador debe recuperar una trama real recibida en bloques IQ."""

from pathlib import Path
import time

import numpy as np
import pytest

from dsp.demoduladores.lora import DemoduladorLoRa


FIXTURE = Path(__file__).parent / "fixtures" / "lora_sf8_500k.npy"


def centered_fixture():
    iq = np.load(FIXTURE).copy()
    iq -= np.mean(iq)
    iq *= np.exp(-2j * np.pi * 500_000 * np.arange(len(iq)) / 2_000_000)
    return iq


def test_live_frequency_plot_preserves_tone_frequency():
    sample_rate = 2_000_000
    tone_hz = 50_000
    iq = np.exp(2j * np.pi * tone_hz * np.arange(32_768) / sample_rate)
    demod = DemoduladorLoRa()
    demod.configurar(sample_rate, 4096, 125_000, 7)
    try:
        result = demod.procesar(iq)
        visual = result['metricas']['lora_visual']
        assert len(visual['time_ms']) == len(visual['magnitude']) == len(visual['freq_khz'])
        assert len(visual['time_ms']) <= 3000
        assert np.median(visual['freq_khz']) == pytest.approx(50.0, abs=0.1)
        assert np.median(visual['magnitude']) == pytest.approx(1.0, abs=0.01)
    finally:
        demod.close()


def test_stream_decodes_frame_and_resets_on_reconfiguration():
    iq = centered_fixture()
    stream = np.concatenate((iq, np.zeros(600_000, dtype=np.complex64)))

    demod = DemoduladorLoRa()
    demod.configurar(2_000_000, 4096, 500_000, 8)
    try:
        for start in range(0, len(stream), 8192):
            demod.procesar(stream[start:start + 8192])

        frame = None
        frame_visual = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = demod.procesar(np.zeros(8192, dtype=np.complex64))
            if result is not None and result.get("metricas", {}).get("lora_frame"):
                frame = result["metricas"]["lora_frame"]
                frame_visual = result["metricas"]["lora_visual"]
                break
            time.sleep(0.02)

        assert frame is not None
        assert frame.payload == b"INTI"
        assert frame.crc_ok is True
        assert frame_visual is not None
        assert frame_visual['complete_frame'] is True
        assert frame_visual['duration_ms'] > 12
        assert frame_visual['time_ms'][-1] > 0.95 * frame_visual['duration_ms']
        assert np.max(frame_visual['magnitude']) > 0
        assert len(frame_visual['freq_khz']) > 0

        demod.configurar(2_000_000, 4096, 125_000, 12)
        assert demod.last_frame is None
        assert demod.bandwidth_hz == 125_000
        assert demod.sf == 12
    finally:
        demod.close()


def test_first_frame_after_idle_is_decoded():
    """Una sola trama tras varios segundos de ruido no debe esperar otra repetición."""
    rng = np.random.default_rng(17)
    demod = DemoduladorLoRa()
    demod.configurar(2_000_000, 4096, 500_000, 8)
    try:
        for _ in range(80):
            noise = (
                rng.normal(0, 0.02, 65_536)
                + 1j * rng.normal(0, 0.02, 65_536)
            ).astype(np.complex64)
            demod.procesar(noise)
        time.sleep(0.3)

        stream = np.concatenate((centered_fixture(), np.zeros(600_000, np.complex64)))
        frame = None
        for start in range(0, len(stream), 8192):
            result = demod.procesar(stream[start:start + 8192])
            if result is not None:
                frame = result.get('metricas', {}).get('lora_frame') or frame

        deadline = time.monotonic() + 5
        while frame is None and time.monotonic() < deadline:
            result = demod.procesar(np.zeros(8192, np.complex64))
            if result is not None:
                frame = result.get('metricas', {}).get('lora_frame')
            time.sleep(0.02)

        assert frame is not None
        assert frame.payload == b'INTI'
        assert frame.crc_ok is True
    finally:
        demod.close()
