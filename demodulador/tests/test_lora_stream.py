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


def test_packet_waterfall_removes_dc_and_keeps_full_iq_band():
    sample_rate = 2_000_000
    t = np.arange(8192) / sample_rate
    iq = (0.25 + np.exp(2j * np.pi * 750_000 * t)).astype(np.complex64)

    waterfall = DemoduladorLoRa._packet_waterfall(iq, sample_rate)
    frequencies = waterfall['freq_hz']
    spectrum = waterfall['power_db'][:, waterfall['power_db'].shape[1] // 2]

    assert waterfall['power_db'].shape[0] == 1024
    assert frequencies[0] == -sample_rate / 2
    assert frequencies[-1] == sample_rate / 2 - sample_rate / 1024
    assert spectrum[np.argmin(abs(frequencies - 750_000))] > np.median(spectrum) + 40
    assert spectrum[np.argmin(abs(frequencies))] < (
        spectrum[np.argmin(abs(frequencies - 750_000))] - 40
    )
    np.testing.assert_allclose(
        waterfall['power_db'],
        DemoduladorLoRa._packet_waterfall(iq - 0.25, sample_rate)['power_db'],
        atol=0.1,
    )


def test_stream_decodes_frame_and_resets_on_reconfiguration():
    iq = centered_fixture()
    stream = np.concatenate((iq, np.zeros(600_000, dtype=np.complex64)))

    demod = DemoduladorLoRa()
    demod.configurar(2_000_000, 4096, 500_000, 8)
    try:
        for start in range(0, len(stream), 8192):
            demod.procesar(stream[start:start + 8192])

        frame = None
        frame_waterfall = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = demod.procesar(np.zeros(8192, dtype=np.complex64))
            if result is not None and result.get("metricas", {}).get("lora_frame"):
                frame = result["metricas"]["lora_frame"]
                frame_waterfall = result["metricas"]["lora_waterfall"]
                break
            time.sleep(0.02)

        assert frame is not None
        assert frame.payload == b"INTI"
        assert frame.crc_ok is True
        assert frame_waterfall['duration_ms'] > 12
        assert frame_waterfall['power_db'].shape[0] == len(frame_waterfall['freq_hz'])
        assert frame_waterfall['power_db'].shape[1] > 1
        assert frame_waterfall['freq_hz'][0] == -1_000_000

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
