"""Grupos RDS y recepción de la subportadora a partir de MPX sintético."""

import numpy as np
import pytest
from time import monotonic, sleep

from dsp.demoduladores.wbfm import DemoduladorWBFM
from dsp.demoduladores.wbfm_audio import DemoduladorWBFMAudio
from dsp.rds import RDSDecoder, _checkword


OFFSETS = (0x0FC, 0x198, 0x168, 0x1B4)


def encode_group(words):
    bits = []
    for index, word in enumerate(words):
        offset = 0x350 if index == 2 and words[1] & 0x0800 else OFFSETS[index]
        transmitted = (word << 10) | (_checkword(word) ^ offset)
        bits.extend((transmitted >> shift) & 1 for shift in range(25, -1, -1))
    return bits


def make_groups():
    pi = 0xA52C
    ps = b"RADIO123"
    rt = b"HOLA ARGENTINA\r".ljust(64, b" ")
    groups = []
    for _ in range(4):
        for segment in range(4):
            pair = ps[2 * segment:2 * segment + 2]
            groups.append((pi, 0x0400 | segment, 0,
                           (pair[0] << 8) | pair[1]))
        for segment in range(4):
            quartet = rt[4 * segment:4 * segment + 4]
            groups.append((pi, 0x2400 | segment,
                           (quartet[0] << 8) | quartet[1],
                           (quartet[2] << 8) | quartet[3]))
    return groups


def make_mpx(bits, fs=300_000, phase=0.67, carrier_offset=0.0):
    bit_rate = (57_000 + carrier_offset) / 48
    duration = len(bits) / bit_rate
    time = np.arange(int(np.ceil(duration * fs))) / fs
    position = time * bit_rate
    bit_index = np.minimum(position.astype(np.int64), len(bits) - 1)
    differentially_encoded = np.bitwise_xor.accumulate(np.asarray(bits, dtype=np.uint8))
    sign = 1 - 2 * differentially_encoded[bit_index].astype(np.int8)
    biphase = np.where(position - np.floor(position) < 0.5, sign, -sign)
    return (2.0 * biphase * np.cos(2 * np.pi * (57_000 + carrier_offset) * time + phase)
            + 5.0 * np.cos(2 * np.pi * 19_000 * time)
            + 8.0 * np.cos(2 * np.pi * 1_000 * time))


def test_rds_recovers_ps_and_radiotext_from_mpx_chunks():
    bits = [bit for group in make_groups() for bit in encode_group(group)]
    mpx = make_mpx(bits)
    decoder = RDSDecoder(300_000)
    for start in range(0, len(mpx), 31_337):
        result = decoder.process(mpx[start:start + 31_337])
    assert result['pi'] == 0xA52C
    assert result['ps'] == 'RADIO123'
    assert result['radiotext'] == 'HOLA ARGENTINA'
    assert result['groups_decoded'] >= 8


def test_rds_recovers_with_carrier_offset_and_mpx_noise():
    bits = [bit for group in make_groups() for bit in encode_group(group)]
    mpx = make_mpx(bits, carrier_offset=5.0)
    mpx += np.random.default_rng(12).normal(0, 1.5, len(mpx))
    decoder = RDSDecoder(300_000)
    for start in range(0, len(mpx), 42_113):
        result = decoder.process(mpx[start:start + 42_113])
    assert result['ps'] == 'RADIO123'
    assert result['radiotext'] == 'HOLA ARGENTINA'


def test_rds_does_not_publish_bad_crc():
    groups = make_groups()[:4]
    bits = [bit for group in groups for bit in encode_group(group)]
    bits[25] ^= 1
    mpx = make_mpx(bits)
    decoder = RDSDecoder(300_000)
    result = decoder.process(mpx)
    assert result['pi'] == 0xA52C
    assert result['ps'] is None


def test_rds_version_b_groups_recover_ps_and_short_radiotext():
    pi = 0xC123
    ps = b"FM TEST "
    rt = b"RDS B\r".ljust(32, b' ')
    groups = []
    for _ in range(3):
        for segment in range(4):
            pair = ps[2 * segment:2 * segment + 2]
            groups.append((pi, 0x0800 | segment, pi,
                           (pair[0] << 8) | pair[1]))
        for segment in range(3):
            pair = rt[2 * segment:2 * segment + 2]
            groups.append((pi, 0x2800 | segment, pi,
                           (pair[0] << 8) | pair[1]))
    bits = [bit for group in groups for bit in encode_group(group)]
    mpx = make_mpx(bits)
    decoder = RDSDecoder(300_000)
    result = decoder.process(mpx)
    assert result['pi'] == pi
    assert result['ps'] == 'FM TEST'
    assert result['radiotext'] == 'RDS B'


@pytest.mark.parametrize('demod_class', [DemoduladorWBFM, DemoduladorWBFMAudio])
def test_both_fm_modes_recover_ps_from_iq(demod_class):
    fs = 2_400_000
    groups = [group for group in make_groups() if group[1] >> 12 == 0]
    bits = [bit for group in groups for bit in encode_group(group)]
    mpx = make_mpx(bits, fs=fs)
    iq = np.exp(2j * np.pi * np.cumsum(mpx * 1_000) / fs).astype(np.complex64)

    demod = demod_class()
    demod.configurar(fs, 4096)
    chunk_size = fs // (2 if demod_class is DemoduladorWBFM else 10)
    result = None
    for start in range(0, len(iq) - chunk_size + 1, chunk_size):
        demod.procesar(iq[start:start + chunk_size])
        deadline = monotonic() + 10
        new_result = None
        while new_result is None and monotonic() < deadline:
            new_result = demod.procesar(np.empty(0, dtype=np.complex64))
            if new_result is None:
                sleep(0.01)
        assert new_result is not None
        result = new_result
    assert result['metricas']['rds']['ps'] == 'RADIO123'
