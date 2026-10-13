"""Vectores de capa de enlace BLE, independientes de la detección IQ."""

import numpy as np
import pytest

from dsp.demoduladores.btle_frames import (
    ADVERTISING_ACCESS_BITS, crc24_ble, decode_legacy_pdu_bits, dewhiten,
)
from dsp.demoduladores.btle import DemoduladorBTLE
from plot_renderer import format_btle_packet_details


def _on_air_bits(pdu, channel):
    packet = pdu + crc24_ble(pdu)
    return dewhiten(np.unpackbits(np.frombuffer(packet, dtype=np.uint8),
                                 bitorder='little'), channel)


def test_crc_24_ble_standard_check_value():
    # Valor check de CRC-24/BLE para la secuencia ASCII 123456789.
    assert crc24_ble(b'123456789') == bytes.fromhex('56 5A C2')
    assert np.array_equal(
        ADVERTISING_ACCESS_BITS[:8],
        np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.uint8),
    )


def test_legacy_advertisement_decodes_on_each_primary_channel():
    uuid_air = bytes.fromhex('22 5B 21 E1 F8 11 56 BF A2 4F 0C 3A A5 8B 85 D1')
    service = bytes([0x19, 0x21]) + uuid_air + b'12345678'
    payload = bytes.fromhex('66 55 44 33 22 11') + service
    pdu = bytes([0x42, len(payload)]) + payload
    for channel in (37, 38, 39):
        decoded = decode_legacy_pdu_bits(_on_air_bits(pdu, channel), channel)
        assert decoded['pdu_type'] == 'ADV_NONCONN_IND'
        assert decoded['advertiser_address'] == '11:22:33:44:55:66'
        assert decoded['tx_address_random'] is True
        assert decoded['crc_valid'] is True
        assert decoded['ad_fields'][0]['uuid'] == 'd1858ba5-3a0c-4fa2-bf56-11f8e1215b22'
        assert decoded['ad_fields'][0]['data_hex'] == b'12345678'.hex(' ').upper()
        assert decode_legacy_pdu_bits(_on_air_bits(pdu, channel),
                                      37 if channel != 37 else 38) is None


def test_corrupt_or_incomplete_advertisement_is_rejected():
    payload = bytes.fromhex('66 55 44 33 22 11')
    pdu = bytes([0x02, len(payload)]) + payload
    bits = _on_air_bits(pdu, 37)
    assert decode_legacy_pdu_bits(bits[:-1], 37) is None
    corrupt = bits.copy()
    corrupt[22] ^= 1
    assert decode_legacy_pdu_bits(corrupt, 37) is None


def test_readable_view_extracts_text_without_interpreting_binary_as_text():
    # Anuncios reales de los JBL Wave Buds 2 capturados en los canales 37/38.
    examples = (
        ('44 1A 31 E3 88 E0 44 6F 13 09 4A 42 4C 20 57 61 76 65 '
         '20 42 75 64 73 20 32 2D 4C 45', 'JBL Wave Buds 2-LE'),
        ('60 25 31 E3 88 E0 44 6F 1E FF 06 00 03 01 80 F6 87 44 '
         '5C 5E 50 04 04 24 4A 42 4C 20 57 61 76 65 20 42 75 '
         '64 73 20 32', 'JBL Wave Buds 2'),
    )
    for pdu_hex, expected_text in examples:
        packet = decode_legacy_pdu_bits(_on_air_bits(bytes.fromhex(pdu_hex), 37), 37)
        packet.update(count=1, time_s=0.1)
        raw = format_btle_packet_details(packet)
        readable = format_btle_packet_details(packet, readable=True)
        assert f'PDU: {pdu_hex}' in raw
        assert f'Texto legible: {expected_text}' in readable
        assert 'PDU: ' not in readable
        assert 'D\\^P' not in readable

    fast_pair = bytes.fromhex(
        '60 10 31 E3 88 E0 44 6F 06 16 2C FE 30 E7 A4 02 0A F5')
    packet = decode_legacy_pdu_bits(_on_air_bits(fast_pair, 37), 37)
    packet.update(count=2, time_s=0.2)
    readable = format_btle_packet_details(packet, readable=True)
    assert 'Texto legible: (sin texto en esta trama)' in readable
    assert 'Google Fast Pair: modelo 0x30E7A4' in readable


def test_lifepulse_packet_survives_early_preamble_correlation_peak():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 4096, bw_mhz=1)
    demod.set_center_frequency(2_402_000_000)
    # Trama LifePulse de la captura real: manual (0x02), ID de seis bytes,
    # presión no disponible (0xFF). Android usó ADV_SCAN_IND no conectable.
    pdu = bytes.fromhex(
        'E6 20 72 4D C5 38 FD 7E 19 21 22 5B 21 E1 F8 11 56 BF '
        'A2 4F 0C 3A A5 8B 85 D1 02 05 4B 48 EE 56 49 FF')
    bits = np.concatenate((
        demod._preamble_bits_variants[0].astype(np.uint8),
        ADVERTISING_ACCESS_BITS,
        _on_air_bits(pdu, 37),
    ))
    true_start = 200
    frequency = np.concatenate((
        np.zeros(true_start), demod._generate_gfsk_freq_reference(bits),
    ))

    packet, found_start = demod._decode_legacy_candidate(
        frequency, true_start - 4 * demod._samples_per_bit)

    assert packet['pdu_type'] == 'ADV_SCAN_IND'
    assert packet['ad_fields'][0]['uuid'] == 'd1858ba5-3a0c-4fa2-bf56-11f8e1215b22'
    assert packet['ad_fields'][0]['data_hex'] == '02 05 4B 48 EE 56 49 FF'
    assert abs(found_start - true_start) < demod._samples_per_bit


@pytest.mark.parametrize('sample_rate, channel, center_hz', [
    (10_000_000, 37, 2_402_000_000),
    (20_000_000, 37, 2_402_000_000),
    (20_000_000, 38, 2_426_000_000),
    (20_000_000, 39, 2_480_000_000),
])
def test_gfsk_iq_recovers_legacy_advertisement(sample_rate, channel, center_hz):
    demod = DemoduladorBTLE()
    demod.configurar(sample_rate, 2048)
    demod.set_center_frequency(center_hz)
    demod.skip_metrics = True
    payload = bytes.fromhex('66 55 44 33 22 11 05 09 54 65 73 74')
    pdu = bytes([0x02, len(payload)]) + payload
    bits = np.concatenate((
        demod._preamble_bits_variants[0].astype(np.uint8),
        ADVERTISING_ACCESS_BITS,
        _on_air_bits(pdu, channel),
    ))
    frequency = demod._generate_gfsk_freq_reference(bits) + 40_000
    burst = np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
    rng = np.random.default_rng(34)
    count = int(demod.sample_rate * demod.buffer_len_s)
    iq = (rng.normal(0, 0.003, count)
          + 1j * rng.normal(0, 0.003, count)).astype(np.complex64)
    start = int(sample_rate * 5e-3)
    iq[start:start + len(burst)] += 0.7 * burst

    metrics = demod.procesar(iq)['metricas']['btle_metrics']

    assert metrics['packet']['pdu_type'] == 'ADV_NONCONN_IND'
    assert metrics['packet']['advertiser_address'] == '11:22:33:44:55:66'
    assert metrics['packet']['ad_fields'][0]['text'] == 'Test'

    invalid = bits.copy()
    invalid[-8] ^= 1  # Romper el CRC en el aire.
    bad_demod = DemoduladorBTLE()
    bad_demod.configurar(sample_rate, 2048)
    bad_demod.set_center_frequency(center_hz)
    bad_frequency = bad_demod._generate_gfsk_freq_reference(invalid) + 40_000
    bad_burst = np.exp(2j * np.pi * np.cumsum(bad_frequency) / bad_demod.sample_rate)
    bad_iq = iq.copy()
    bad_iq[start:start + len(bad_burst)] = 0.7 * bad_burst
    assert bad_demod.procesar(bad_iq)['metricas']['btle_metrics'] is None


def test_overlap_does_not_count_the_same_advertisement_twice():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    demod.set_center_frequency(2_402_000_000)
    demod.skip_metrics = True
    payload = bytes.fromhex('66 55 44 33 22 11')
    pdu = bytes([0x02, len(payload)]) + payload
    bits = np.concatenate((
        demod._preamble_bits_variants[0].astype(np.uint8),
        ADVERTISING_ACCESS_BITS,
        _on_air_bits(pdu, 37),
    ))
    burst = np.exp(2j * np.pi * np.cumsum(
        demod._generate_gfsk_freq_reference(bits)) / demod.sample_rate)
    block = int(demod.sample_rate * demod.buffer_len_s)
    iq = np.zeros(2 * block, dtype=np.complex64)
    start = block - int(2e-3 * demod.sample_rate)
    iq[start:start + len(burst)] = 0.7 * burst

    first = demod.procesar(iq[:block])
    second = demod.procesar(iq[block:])

    assert first['metricas']['btle_metrics']['packet']['count'] == 1
    assert second['metricas']['btle_metrics'] is None
    assert demod.valid_packet_count == 1


def test_two_advertisements_in_one_window_are_both_counted():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    demod.set_center_frequency(2_402_000_000)
    demod.skip_metrics = True
    payload = bytes.fromhex('66 55 44 33 22 11')
    pdu = bytes([0x02, len(payload)]) + payload
    bits = np.concatenate((
        demod._preamble_bits_variants[0].astype(np.uint8),
        ADVERTISING_ACCESS_BITS,
        _on_air_bits(pdu, 37),
    ))
    burst = np.exp(2j * np.pi * np.cumsum(
        demod._generate_gfsk_freq_reference(bits)) / demod.sample_rate)
    count = int(demod.sample_rate * demod.buffer_len_s)
    iq = np.zeros(count, dtype=np.complex64)
    for start in (100_000, 300_000):
        iq[start:start + len(burst)] = 0.7 * burst

    metrics = demod.procesar(iq)['metricas']['btle_metrics']

    assert [packet['count'] for packet in metrics['packets']] == [1, 2]
    assert metrics['packet']['count'] == 2
    assert demod.valid_packet_count == 2
