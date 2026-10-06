"""El adaptador debe recuperar una trama real recibida en bloques IQ."""

from pathlib import Path
import time

import numpy as np

from dsp.demoduladores.lora import DemoduladorLoRa
from dsp.demoduladores.lora_core import (
    LoRaConfig, _chirp_references, _hamming_codeword, _header_checksum,
    _payload_crc, _whitening_sequence, decode_capture,
)


FIXTURE = Path(__file__).parent / "fixtures" / "lora_sf8_500k.npy"


def centered_fixture():
    iq = np.load(FIXTURE).copy()
    iq -= np.mean(iq)
    iq *= np.exp(-2j * np.pi * 500_000 * np.arange(len(iq)) / 2_000_000)
    return iq


def _interleave_lora_block(nibbles, coding_rate):
    """Inversa de la operación usada por el receptor para crear un test RF."""
    width = len(nibbles)
    symbols_per_block = coding_rate + 4
    codewords = [_hamming_codeword(nibble, coding_rate) for nibble in nibbles]
    symbols = []
    for symbol_index in range(symbols_per_block):
        gray = 0
        for bit_index in range(width):
            codeword = codewords[(symbol_index - bit_index - 1) % width]
            bit = (codeword >> (symbols_per_block - 1 - symbol_index)) & 1
            gray |= bit << (width - 1 - bit_index)
        value = gray
        for shift in (1, 2, 4, 8):
            value ^= value >> shift
        symbols.append(value)
    return symbols


def synthetic_sf12_125k_packet():
    """Trama con LDRO y CR 4/8; dura más que la vieja ventana de un segundo."""
    payload = b'TEST LORA 125k SF12'
    config = LoRaConfig(125_000, 12)
    samples_per_symbol = config.samples_per_symbol
    up, down = (reference.astype(np.complex64)
                for reference in _chirp_references(config))
    sample_index = np.arange(samples_per_symbol)

    def shifted_up(bin_index):
        tone = np.exp(2j * np.pi * bin_index * sample_index / samples_per_symbol)
        return up * tone.astype(np.complex64)

    crc = _payload_crc(payload)
    encoded_bytes = [byte ^ mask for byte, mask in
                     zip(payload, _whitening_sequence(len(payload)))]
    encoded_bytes += [crc & 0xFF, crc >> 8]
    data_nibbles = [nibble for byte in encoded_bytes
                    for nibble in (byte & 15, byte >> 4)]
    n0, n1, n2 = len(payload) >> 4, len(payload) & 15, 4 * 2 + 1
    checksum = _header_checksum(n0, n1, n2)
    header_nibbles = [n0, n1, n2, checksum >> 4, checksum & 15] + data_nibbles[:5]
    header_symbols = _interleave_lora_block(header_nibbles, 4)
    payload_symbols = []
    remaining = data_nibbles[5:]
    for start in range(0, len(remaining), config.sf - 2):
        block = remaining[start:start + config.sf - 2]
        block += [0] * (config.sf - 2 - len(block))
        payload_symbols.extend(_interleave_lora_block(block, 4))

    parts = [np.zeros(2 * samples_per_symbol, np.complex64)]
    parts += [up] * 8 + [shifted_up(24), shifted_up(32)] + [down] * 2
    parts += [down[:samples_per_symbol // 4]]
    parts += [shifted_up(4 * symbol + 1) for symbol in header_symbols]
    parts += [shifted_up(4 * symbol + 1) for symbol in payload_symbols]
    parts += [np.zeros(2 * samples_per_symbol, np.complex64)]
    return np.concatenate(parts), payload, config


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


def test_sf12_125k_stream_keeps_preamble_until_packet_is_complete():
    iq, payload, config = synthetic_sf12_125k_packet()
    offline = decode_capture(iq, config)
    assert offline.payload == payload
    assert offline.crc_ok is True

    demod = DemoduladorLoRa()
    demod.configurar(2_000_000, 4096, 125_000, 12)
    received = []
    chunk_size = 32_768
    try:
        for source in (np.zeros(1_000_000, np.complex64), iq):
            for start in range(0, len(source), chunk_size):
                chunk = source[start:start + chunk_size]
                result = demod.procesar(chunk)
                if result is not None:
                    frame = result.get('metricas', {}).get('lora_frame')
                    if frame is not None:
                        received.append(frame)
                time.sleep(len(chunk) / config.sample_rate)

        deadline = time.monotonic() + 8
        while not received and time.monotonic() < deadline:
            result = demod.procesar(np.zeros(chunk_size, np.complex64))
            if result is not None:
                frame = result.get('metricas', {}).get('lora_frame')
                if frame is not None:
                    received.append(frame)
            time.sleep(0.02)

        assert len(received) == 1
        assert received[0].payload == payload
        assert received[0].crc_ok is True
        assert received[0].sync_word == 0x34
        assert received[0].coding_rate == 4
    finally:
        demod.close()
