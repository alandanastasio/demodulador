"""Decodificación RDS desde el multiplex FM demodulado.

RDS usa una subportadora suprimida de 57 kHz, biphase a 1187,5 bit/s,
codificación diferencial y grupos de cuatro bloques de 26 bits. Las
decisiones diferenciales entre símbolos complejos evitan tener que fijar
la fase absoluta de la subportadora. Solo se publican grupos con los cuatro
checkwords correctos.
"""

from fractions import Fraction

import numpy as np

from .streaming_resampler import StreamingResampler


_OFFSETS = (0x0FC, 0x198, 0x168, 0x1B4)
_OFFSET_C_PRIME = 0x350
_CODEWORD_MASK = (1 << 26) - 1


def _checkword(data):
    remainder = 0
    for shift in range(15, -1, -1):
        feedback = ((remainder >> 9) ^ (data >> shift)) & 1
        remainder = (remainder << 1) & 0x3FF
        if feedback:
            remainder ^= 0x1B9
    return remainder


def _block_data(codeword, offset):
    data = codeword >> 10
    return data if (_checkword(data) ^ offset) == (codeword & 0x3FF) else None


class _BitFramer:
    def __init__(self):
        self.codeword = 0
        self.bit_index = 0
        self.candidates = []

    def feed(self, bit):
        self.codeword = ((self.codeword << 1) | int(bit)) & _CODEWORD_MASK
        self.bit_index += 1
        groups = []
        remaining = []
        for next_bit, words in self.candidates:
            if self.bit_index != next_bit:
                remaining.append((next_bit, words))
                continue
            position = len(words)
            offset = (_OFFSET_C_PRIME if position == 2 and words[1] & 0x0800
                      else _OFFSETS[position])
            data = _block_data(self.codeword, offset)
            if data is None:
                continue
            completed = words + (data,)
            if position == 3:
                groups.append(completed)
            else:
                remaining.append((next_bit + 26, completed))
        self.candidates = remaining
        if self.bit_index >= 26:
            first = _block_data(self.codeword, _OFFSETS[0])
            if first is not None:
                self.candidates.append((self.bit_index + 26, (first,)))
        return groups


_RDS_G0_TABLE = (
    " !\"#¤%&'()*+,-./"
    "0123456789:;<=>?"
    "@ABCDEFGHIJKLMNO"
    "PQRSTUVWXYZ[\\]―_"
    "‖abcdefghijklmno"
    "pqrstuvwxyz{|}¯ "
    "áàéèíìóòúùÑÇŞß¡Ĳ"
    "âäêëîïôöûüñçşğıĳ"
    "ªα©‰Ğěňőπ€£$←↑→↓"
    "º¹²³±İńűµ¿÷°¼½¾§"
    "ÁÀÉÈÍÌÓÒÚÙŘČŠŽĐĿ"
    "ÂÄÊËÎÏÔÖÛÜřčšžđŀ"
    "ÃÅÆŒŷÝÕØÞŊŔĆŚŹŦð"
    "ãåæœŵýõøþŋŕćśźŧ "
)


def _rds_text(raw):
    """Convierte bytes del repertorio RDS básico (IEC 62106 Anexo E, tabla G0)."""
    payload = raw.split(b'\r', 1)[0]
    chars = []
    for byte in payload:
        if byte >= 0x20:
            chars.append(_RDS_G0_TABLE[byte - 0x20])
        elif byte in (0x0A, 0x0B):
            chars.append(' ')
    return ''.join(chars).strip()



class RDSDecoder:
    """Recibe bloques MPX de una misma estación y conserva PS/RadioText."""

    def __init__(self, sample_rate):
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 130_000:
            raise ValueError('El multiplex RDS requiere al menos 130 kS/s.')
        # Limitar el denominador acota el FIR polifásico para tasas de captura
        # no estándar sin afectar la sincronización de 1187,5 bit/s.
        ratio = Fraction(9_500 / self.sample_rate).limit_denominator(8_192)
        self.resampler = StreamingResampler(ratio.numerator, ratio.denominator)
        self._carrier_phase = 0.0
        self._baseband = np.empty(0, dtype=np.complex128)
        self._baseband_start = 0
        self._next_symbols = list(range(8))
        self._previous_symbols = [None] * 8
        self._framers = [_BitFramer() for _ in range(8)]
        self.pi = None
        self.pty = None
        self.tp = None
        self.ta = None
        self.ps = None
        self.radiotext = None
        self._ps_segments = [None] * 4
        self._rt_segments = [None] * 16
        self._rt_ab = None
        self._rt_version = None
        self.groups_decoded = 0

    def process(self, multiplex):
        multiplex = np.asarray(multiplex)
        if not len(multiplex):
            return self.snapshot()
        step = 2 * np.pi * 57_000 / self.sample_rate
        phase = self._carrier_phase + step * np.arange(len(multiplex))
        mixed = multiplex * np.exp(-1j * phase)
        self._carrier_phase = (self._carrier_phase + step * len(multiplex)) % (2 * np.pi)
        baseband = self.resampler.process(mixed)
        if not len(baseband):
            return self.snapshot()
        self._baseband = np.concatenate((self._baseband, baseband))
        end = self._baseband_start + len(self._baseband)
        for phase_index in range(8):
            next_symbol = self._next_symbols[phase_index]
            previous = self._previous_symbols[phase_index]
            framer = self._framers[phase_index]
            while next_symbol + 8 <= end:
                start = next_symbol - self._baseband_start
                chip_pair = self._baseband[start + 1:start + 4].mean() - (
                    self._baseband[start + 5:start + 8].mean())
                if previous is not None:
                    bit = (chip_pair * np.conj(previous)).real < 0
                    for group in framer.feed(bit):
                        self._accept_group(group)
                previous = chip_pair
                next_symbol += 8
            self._next_symbols[phase_index] = next_symbol
            self._previous_symbols[phase_index] = previous
        discard = min(self._next_symbols) - self._baseband_start
        self._baseband = self._baseband[discard:].copy()
        self._baseband_start += discard
        return self.snapshot()

    def _accept_group(self, group):
        pi, block_b, block_c, block_d = group
        if self.pi != pi:
            self.pi = pi
            self.ps = None
            self.radiotext = None
            self._ps_segments = [None] * 4
            self._rt_segments = [None] * 16
            self._rt_ab = None
            self._rt_version = None
        self.groups_decoded += 1
        group_type = (block_b >> 12) & 0xF
        version = (block_b >> 11) & 1
        self.tp = bool(block_b & 0x0400)
        self.pty = (block_b >> 5) & 0x1F
        if group_type == 0:
            self.ta = bool(block_b & 0x0010)
            segment = block_b & 0x3
            self._ps_segments[segment] = bytes((block_d >> 8, block_d & 0xFF))
            if all(part is not None for part in self._ps_segments):
                self.ps = _rds_text(b''.join(self._ps_segments))
        elif group_type == 2:
            ab = (block_b >> 4) & 1
            if ab != self._rt_ab or version != self._rt_version:
                self._rt_segments = [None] * 16
                self.radiotext = None
                self._rt_ab = ab
                self._rt_version = version
            segment = block_b & 0xF
            payload = (bytes((block_c >> 8, block_c & 0xFF,
                              block_d >> 8, block_d & 0xFF)) if version == 0
                       else bytes((block_d >> 8, block_d & 0xFF)))
            self._rt_segments[segment] = payload
            contiguous = []
            for part in self._rt_segments:
                if part is None:
                    break
                contiguous.append(part)
                if b'\r' in part:
                    self.radiotext = _rds_text(b''.join(contiguous))
                    break
            else:
                if len(contiguous) == 16:
                    self.radiotext = _rds_text(b''.join(contiguous))

    def snapshot(self):
        return {
            'pi': self.pi,
            'ps': self.ps,
            'radiotext': self.radiotext,
            'pty': self.pty,
            'tp': self.tp,
            'ta': self.ta,
            'groups_decoded': self.groups_decoded,
        }
