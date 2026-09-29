"""Regresión con un paquete RF real recortado de la captura SF8 del notebook.

El fixture proviene de lora_20260921_175914_956424.npz, muestras
204234:240330. Contiene una trama con payload INTI y CRC PHY válido.
"""

from pathlib import Path

import numpy as np

from dsp.demoduladores.lora_core import LoRaConfig, decode_capture
from dsp.demoduladores import lora_core


FIXTURE = Path(__file__).parent / "fixtures" / "lora_sf8_500k.npy"


def test_decodes_recorded_lora_frame():
    iq = np.load(FIXTURE)
    config = LoRaConfig(bandwidth_hz=500_000, sf=8, channel_offset_hz=500_000)

    frame = decode_capture(iq, config)

    assert frame.payload == b"INTI"
    assert frame.crc_ok is True
    assert frame.sync_word == 0x34
    assert frame.coding_rate == 1
    assert frame.header_checksum_ok is True


def test_decodes_same_frame_when_channel_is_centered():
    iq = np.load(FIXTURE).copy()
    iq -= np.mean(iq)
    iq *= np.exp(-2j * np.pi * 500_000 * np.arange(len(iq)) / 2_000_000)

    frame = decode_capture(iq, LoRaConfig(bandwidth_hz=500_000, sf=8))

    assert frame.payload == b"INTI"
    assert frame.crc_ok is True


def test_reports_payload_crc_failure(monkeypatch):
    iq = np.load(FIXTURE)
    config = LoRaConfig(bandwidth_hz=500_000, sf=8, channel_offset_hz=500_000)
    monkeypatch.setattr(lora_core, '_payload_crc', lambda payload: -1)

    frame = decode_capture(iq, config)

    assert frame.payload == b'INTI'
    assert frame.has_crc is True
    assert frame.header_checksum_ok is True
    assert frame.crc_ok is False
