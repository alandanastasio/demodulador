"""Decodificación LoRa de una captura IQ, independiente de la SDR y de Qt.

Las etapas siguen captura_y_demod_LORA.ipynb: detección del preámbulo,
sincronización CFO/STO, plegado de la FFT, header explícito y payload PHY.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft as scipy_fft


class LoRaDecodeError(ValueError):
    """La captura no contiene una trama LoRa válida para esta configuración."""


class LoRaIncompleteFrame(LoRaDecodeError):
    """Hay una trama sincronizada, pero aún faltan muestras para terminarla."""

    def __init__(self, required_samples: int, frame_start_sample: int):
        super().__init__("Faltan muestras para completar la trama LoRa.")
        self.required_samples = required_samples
        self.frame_start_sample = frame_start_sample


@dataclass(frozen=True)
class LoRaConfig:
    bandwidth_hz: int
    sf: int
    sample_rate: int = 2_000_000
    channel_offset_hz: float = 0.0
    ldro: bool | None = None

    def __post_init__(self):
        if not 7 <= self.sf <= 12:
            raise ValueError("LoRa requiere SF7 a SF12.")
        if self.bandwidth_hz <= 0 or self.sample_rate <= 0:
            raise ValueError("BW y sample rate deben ser positivos.")
        ratio = self.sample_rate / self.bandwidth_hz
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("El sample rate debe ser un múltiplo entero del BW.")
        if abs(self.channel_offset_hz) + self.bandwidth_hz / 2 >= self.sample_rate / 2:
            raise ValueError("El canal LoRa queda fuera de la banda de Nyquist.")

    @property
    def n_bins(self) -> int:
        return 1 << self.sf

    @property
    def oversampling(self) -> int:
        return round(self.sample_rate / self.bandwidth_hz)

    @property
    def samples_per_symbol(self) -> int:
        return self.n_bins * self.oversampling

    @property
    def symbol_duration(self) -> float:
        return self.n_bins / self.bandwidth_hz

    @property
    def ldro_enabled(self) -> bool:
        # La opción explícita permite reproducir transmisores que no usan el
        # ajuste automático habitual para símbolos de más de 16 ms.
        return self.ldro if self.ldro is not None else self.symbol_duration > 0.016


@dataclass(frozen=True)
class LoRaFrame:
    payload: bytes
    crc_ok: bool | None
    header_checksum_ok: bool
    sync_word: int
    coding_rate: int
    has_crc: bool
    cfo_hz: float
    timing_correction_samples: int
    drift_bins_per_symbol: float
    coarse_start_sample: int
    frame_start_sample: int
    sfd_start_sample: int
    header_start_sample: int
    payload_start_sample: int
    observed_bins: tuple[int | None, ...]
    observed_offsets: tuple[int, ...]
    header_bins: tuple[int, ...]
    payload_bins: tuple[int, ...]
    header_symbols: tuple[int, ...]
    payload_symbols: tuple[int, ...]
    header_concentrations: tuple[float, ...]
    payload_concentrations: tuple[float, ...]
    folded_power: np.ndarray = field(compare=False, repr=False)
    folded_section_starts: tuple[int, int, int]


def instantaneous_frequency(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    """Frecuencia instantánea en Hz; útil para el gráfico de chirps."""
    samples = np.asarray(iq)
    if samples.ndim != 1 or not np.iscomplexobj(samples):
        raise ValueError("Se requiere un vector IQ complejo.")
    return np.angle(samples[1:] * np.conjugate(samples[:-1])) * (sample_rate / (2 * np.pi))


def _prepare_iq(iq_raw: np.ndarray, config: LoRaConfig) -> np.ndarray:
    dc = np.mean(iq_raw, dtype=np.complex128)
    iq_base = np.empty(len(iq_raw), dtype=np.complex64)
    for begin in range(0, len(iq_raw), 262_144):
        end = min(begin + 262_144, len(iq_raw))
        chunk = iq_raw[begin:end] - dc
        if config.channel_offset_hz:
            chunk = chunk * np.exp(
                -2j * np.pi * config.channel_offset_hz
                * np.arange(begin, end) / config.sample_rate
            )
        iq_base[begin:end] = chunk
    scale = float(np.std(iq_base))
    if not np.isfinite(scale) or scale <= 0:
        raise LoRaDecodeError("La captura IQ está vacía o no tiene variación.")
    iq_base /= scale
    return iq_base


def _chirp_references(config: LoRaConfig) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(config.samples_per_symbol) / config.sample_rate
    phase = 2 * np.pi * (
        -config.bandwidth_hz / 2 * t
        + config.bandwidth_hz / (2 * config.symbol_duration) * t**2
    )
    up = np.exp(1j * phase)
    return up, np.conjugate(up)


def _circular_distance(a: float, b: float, size: int) -> float:
    return abs((a - b + size / 2) % size - size / 2)


def _chirp_fft(samples: np.ndarray, reference: np.ndarray, interpolation: int = 1) -> tuple[float, float]:
    if interpolation > 1 and samples.dtype == np.complex64:
        # La interpolación 32x crea FFT muy grandes en SF12/BW125. SciPy
        # conserva complex64 y evita el costo de promoverlas a complex128.
        # Las entradas de doble precisión mantienen la ruta original.
        dechirped = np.multiply(samples, reference, dtype=np.complex64)
        spectrum = scipy_fft(dechirped, n=len(samples) * interpolation)
    else:
        spectrum = np.fft.fft(samples * reference, n=len(samples) * interpolation)
    power = np.abs(spectrum) ** 2
    index = int(np.argmax(power))
    signed_bin = ((index + len(power) // 2) % len(power) - len(power) // 2) / interpolation
    energy = max(len(samples) * np.vdot(samples, samples).real, 1e-30)
    return signed_bin, float(power[index] / energy)


def _folded_power(samples: np.ndarray, reference: np.ndarray, config: LoRaConfig) -> np.ndarray:
    power = np.abs(np.fft.fft(samples * reference)) ** 2
    return power.reshape(config.oversampling, config.n_bins).sum(axis=0)


def _detect_preamble(iq: np.ndarray, down: np.ndarray, config: LoRaConfig,
                     start_sample: int) -> tuple[int, int]:
    n_sym = config.samples_per_symbol
    stride = n_sym // 4
    first = max(0, (start_sample + stride - 1) // stride)
    windows = 1 + (len(iq) - n_sym) // stride
    papr = np.zeros(windows - first)
    bins = np.zeros(windows - first, dtype=int)
    for i in range(first, windows):
        spectrum_power = np.abs(np.fft.fft(iq[i * stride:i * stride + n_sym] * down)) ** 2
        papr[i - first] = spectrum_power.max() / max(float(spectrum_power.mean()), 1e-30)
        bins[i - first] = int(np.argmax(spectrum_power))
        if i - first < 12:
            continue
        candidate = i - 12
        positions = [candidate + 4 * k - first for k in range(4)]
        anchor = bins[positions[0]]
        if all(papr[p] >= 10.0 and _circular_distance(bins[p], anchor, n_sym) <= 2
               for p in positions):
            return candidate * stride, int(anchor)
    raise LoRaDecodeError("No se detectó un preámbulo LoRa para el BW y SF elegidos.")


def _deinterleave(symbols: list[int], sf_app: int) -> list[int]:
    words = [[0] * len(symbols) for _ in range(sf_app)]
    for i, symbol in enumerate(symbols):
        gray = int(symbol) ^ (int(symbol) >> 1)
        for j in range(sf_app):
            words[(i - j - 1) % sf_app][i] = (gray >> (sf_app - 1 - j)) & 1
    return [sum(bit << (len(row) - 1 - i) for i, bit in enumerate(row)) for row in words]


def _hamming_codeword(nibble: int, cr: int) -> int:
    d0, d1, d2, d3 = [(nibble >> k) & 1 for k in range(4)]
    parity = ([d0 ^ d1 ^ d2 ^ d3] if cr == 1 else
              [d0 ^ d1 ^ d2, d1 ^ d2 ^ d3, d0 ^ d1 ^ d3, d0 ^ d2 ^ d3][:cr])
    bits = [d0, d1, d2, d3] + parity
    return sum(bit << (len(bits) - 1 - i) for i, bit in enumerate(bits))


def _decode_hamming(words: list[int], cr: int) -> tuple[list[int], list[int]]:
    if cr not in (1, 2, 3, 4):
        raise LoRaDecodeError("Coding rate LoRa inválido.")
    candidates = [_hamming_codeword(nibble, cr) for nibble in range(16)]
    nibbles, distances = [], []
    for word in words:
        ds = [(int(word) ^ candidate).bit_count() for candidate in candidates]
        distance = min(ds)
        if ds.count(distance) != 1 or distance > (1 if cr >= 3 else 0):
            raise LoRaDecodeError(f"FEC no corregible: palabra 0x{word:X}.")
        nibbles.append(ds.index(distance))
        distances.append(distance)
    return nibbles, distances


def _header_checksum(n0: int, n1: int, n2: int) -> int:
    bit = lambda value, k: (value >> k) & 1
    c4 = bit(n0, 3) ^ bit(n0, 2) ^ bit(n0, 1) ^ bit(n0, 0)
    c3 = bit(n0, 3) ^ bit(n1, 3) ^ bit(n1, 2) ^ bit(n1, 1) ^ bit(n2, 0)
    c2 = bit(n0, 2) ^ bit(n1, 3) ^ bit(n1, 0) ^ bit(n2, 3) ^ bit(n2, 1)
    c1 = bit(n0, 1) ^ bit(n1, 2) ^ bit(n1, 0) ^ bit(n2, 2) ^ bit(n2, 1) ^ bit(n2, 0)
    c0 = bit(n0, 0) ^ bit(n1, 1) ^ bit(n2, 3) ^ bit(n2, 2) ^ bit(n2, 1) ^ bit(n2, 0)
    return (c4 << 4) | (c3 << 3) | (c2 << 2) | (c1 << 1) | c0


def _whitening_sequence(length: int) -> list[int]:
    state = 0xFF
    sequence = []
    for _ in range(length):
        sequence.append(state)
        feedback = ((state >> 7) ^ (state >> 5) ^ (state >> 4) ^ (state >> 3)) & 1
        state = ((state << 1) & 0xFF) | feedback
    return sequence


def _payload_crc(data: bytes) -> int:
    if len(data) == 1:
        return data[0]
    crc = 0
    for byte in data[:-2]:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xFFFF
    return crc ^ data[-1] ^ (data[-2] << 8)


def decode_capture(iq_raw: np.ndarray, config: LoRaConfig, start_sample: int = 0) -> LoRaFrame:
    """Devuelve la primera trama válida de una captura con header explícito."""
    iq_raw = np.asarray(iq_raw)
    if iq_raw.ndim != 1 or not np.iscomplexobj(iq_raw):
        raise ValueError("Se requiere una captura IQ compleja unidimensional.")
    if start_sample < 0 or start_sample >= len(iq_raw):
        raise ValueError("start_sample queda fuera de la captura.")
    n_sym = config.samples_per_symbol
    if len(iq_raw) < 10 * n_sym:
        raise LoRaDecodeError("La captura no alcanza para una trama LoRa.")
    iq = _prepare_iq(iq_raw, config)
    up, down = _chirp_references(config)
    coarse_start, detected_bin = _detect_preamble(iq, down, config, start_sample)

    scan = []
    for k in range(1, 69):
        idx = coarse_start + k * n_sym
        if idx + n_sym > len(iq):
            break
        chunk = iq[idx:idx + n_sym]
        ku, qu = _chirp_fft(chunk, down)
        kd, qd = _chirp_fft(chunk, up)
        scan.append((k, ku, qu, kd, qd))
    sfd_candidates = [i for i, row in enumerate(scan)
                      if row[4] > 0.03 and row[4] > 4 * row[2]]
    if not sfd_candidates:
        raise LoRaDecodeError("No se encontró el downchirp del SFD.")
    sfd_scan = sfd_candidates[0]
    down_row = scan[sfd_scan]
    up_rows = [row for row in scan[:max(0, sfd_scan - 2)]
               if row[2] > 0.03 and row[2] > 3 * row[4]
               and _circular_distance(row[1], detected_bin, n_sym) <= 2]
    if len(up_rows) < 3:
        raise LoRaDecodeError("Faltan upchirps para estimar CFO y STO.")

    up_bins = []
    phase_pairs = []
    for row in up_rows:
        idx = coarse_start + row[0] * n_sym
        up_bins.append(_chirp_fft(iq[idx:idx + n_sym], down, interpolation=32)[0])
    for left, right in zip(up_rows, up_rows[1:]):
        if right[0] == left[0] + 1:
            a = coarse_start + left[0] * n_sym
            cross = np.vdot(iq[a:a + n_sym], iq[a + n_sym:a + 2 * n_sym])
            if abs(cross) > 0:
                phase_pairs.append(cross / abs(cross))
    if not phase_pairs:
        raise LoRaDecodeError("Faltan pares de preámbulo para el CFO fraccional.")
    ku = float(np.median(up_bins))
    down_idx = coarse_start + down_row[0] * n_sym
    kd = _chirp_fft(iq[down_idx:down_idx + n_sym], up, interpolation=32)[0]
    fractional_cfo_bin = np.angle(np.sum(phase_pairs)) / (2 * np.pi)
    solutions = []
    for branch_up in (-config.n_bins, 0, config.n_bins):
        for branch_down in (-config.n_bins, 0, config.n_bins):
            cfo_pair = (ku + branch_up + kd + branch_down) / 2
            cfo_bins = round(cfo_pair - fractional_cfo_bin) + fractional_cfo_bin
            timing = (ku + branch_up - cfo_bins) * config.oversampling
            if abs(cfo_bins) < config.n_bins / 4 and abs(timing) <= n_sym / 2:
                solutions.append((abs(cfo_pair - cfo_bins), cfo_bins, timing))
    if not solutions:
        raise LoRaDecodeError("CFO/STO fuera del rango admitido.")
    _, cfo_bins, timing_samples = min(solutions)
    cfo_hz = cfo_bins / config.symbol_duration
    true_start = coarse_start - int(round(timing_samples))
    if true_start < 0:
        raise LoRaDecodeError("El preámbulo comienza antes de la captura.")

    iq_synced = np.empty(len(iq), dtype=np.complex64)
    for begin in range(0, len(iq), 262_144):
        end = min(begin + 262_144, len(iq))
        iq_synced[begin:end] = iq[begin:end] * np.exp(
            -2j * np.pi * cfo_hz * np.arange(begin, end) / config.sample_rate
        )

    rows = []
    for k in range(-1, min(69, (len(iq_synced) - true_start) // n_sym)):
        idx = true_start + k * n_sym
        if idx < 0:
            rows.append((k, None, 0.0, 0.0))
            continue
        chunk = iq_synced[idx:idx + n_sym]
        pu = _folded_power(chunk, down, config)
        pd = _folded_power(chunk, up, config)
        energy = max(float(n_sym * np.vdot(chunk, chunk).real), 1e-30)
        rows.append((k, int(np.argmax(pu)), float(pu.max() / energy),
                     float(pd.max() / energy)))

    frame_candidate = None
    for j in range(6, len(rows) - 1):
        if not all(r[3] > 0.1 and r[3] > 4 * r[2] for r in rows[j:j + 2]):
            continue
        preamble_rows = rows[j - 6:j - 2]
        if not all(r[1] is not None and r[2] > 0.1 and r[2] > 4 * r[3]
                   for r in preamble_rows):
            continue
        anchor = preamble_rows[0][1]
        if not all(_circular_distance(r[1], anchor, config.n_bins) <= 1
                   for r in preamble_rows):
            continue
        signed = [((r[1] + config.n_bins // 2) % config.n_bins) - config.n_bins // 2
                  for r in preamble_rows]
        residual_bin = int(round(float(np.median(signed))))
        sync_symbols = [(rows[j - 2 + m][1] - residual_bin) % config.n_bins
                        for m in (0, 1)]
        sync_nibbles = [int(round(s / 8)) for s in sync_symbols]
        if not all(0 <= nib <= 15 and
                   _circular_distance(sym, 8 * nib, config.n_bins) <= 1
                   for sym, nib in zip(sync_symbols, sync_nibbles)):
            continue
        frame_candidate = (j, residual_bin, sync_nibbles)
        break
    if frame_candidate is None:
        raise LoRaDecodeError("No se validó preámbulo, sync word y SFD.")
    j, preamble_bin_offset, sync_nibbles = frame_candidate
    sync_word = (sync_nibbles[0] << 4) | sync_nibbles[1]
    sfd_start = true_start + rows[j][0] * n_sym
    header_start = sfd_start + 2 * n_sym + n_sym // 4
    payload_start = header_start + 8 * n_sym

    pilot_rows = [r for r in rows[:j - 2]
                  if r[1] is not None and r[2] > 0.1 and r[2] > 4 * r[3]
                  and _circular_distance(r[1], preamble_bin_offset, config.n_bins) <= 1]
    if len(pilot_rows) < 4:
        raise LoRaDecodeError("Faltan upchirps para medir la deriva residual.")
    pilot_offsets = np.array([r[0] for r in pilot_rows], dtype=float)
    pilot_bins = np.array([
        _chirp_fft(iq_synced[true_start + int(k) * n_sym:
                             true_start + (int(k) + 1) * n_sym], down, interpolation=32)[0]
        for k in pilot_offsets
    ])
    pilot_reference_offset = float(np.mean(pilot_offsets))
    drift, pilot_frequency = np.polyfit(pilot_offsets - pilot_reference_offset, pilot_bins, 1)
    symbol_samples = np.arange(n_sym)

    def folded_power_at(idx: int) -> np.ndarray:
        symbol_offset = (idx - true_start) / n_sym
        residual = pilot_frequency + drift * (symbol_offset - pilot_reference_offset)
        corrected = iq_synced[idx:idx + n_sym] * np.exp(
            -2j * np.pi * residual * symbol_samples / n_sym
        )
        return _folded_power(corrected, down, config)

    def demod_symbols(start: int, count: int, reduced: bool = False):
        if start < 0 or start + count * n_sym > len(iq_synced):
            raise LoRaIncompleteFrame(start + count * n_sym, true_start)
        symbols, peaks, concentrations, powers = [], [], [], []
        for i in range(count):
            idx = start + i * n_sym
            power = folded_power_at(idx)
            peak = int(np.argmax(power))
            value = (peak - 1) % config.n_bins
            symbols.append(value // 4 if reduced else value)
            peaks.append(peak)
            concentrations.append(float(power.max() / max(power.sum(), 1e-30)))
            powers.append(power.astype(np.float32))
        return symbols, peaks, concentrations, powers

    header_symbols, header_bins, header_concentrations, header_powers = demod_symbols(
        header_start, 8, reduced=True
    )
    header_codewords = _deinterleave(header_symbols, config.sf - 2)
    header_nibbles, _ = _decode_hamming(header_codewords, 4)
    n0, n1, n2, n3, n4 = header_nibbles[:5]
    header_checksum = ((n3 & 1) << 4) | n4
    length, cr, has_crc = (n0 << 4) | n1, n2 >> 1, bool(n2 & 1)
    header_checksum_ok = header_checksum == _header_checksum(n0, n1, n2)
    if (not header_checksum_ok or
            not 1 <= cr <= 4 or n3 & 0xE or length == 0):
        raise LoRaDecodeError("El header PHY no pasó el checksum.")

    sf_app = config.sf - 2 if config.ldro_enabled else config.sf
    codeword_len = cr + 4
    nibbles_needed = 2 * length + (4 if has_crc else 0)
    nibbles = list(header_nibbles[5:])
    remaining = max(0, nibbles_needed - len(nibbles))
    n_blocks = (remaining + sf_app - 1) // sf_app
    n_symbols = n_blocks * codeword_len
    payload_symbols, payload_bins, payload_concentrations, payload_powers = demod_symbols(
        payload_start, n_symbols, reduced=config.ldro_enabled
    )
    for block_index in range(n_blocks):
        block = payload_symbols[block_index * codeword_len:(block_index + 1) * codeword_len]
        decoded, _ = _decode_hamming(_deinterleave(block, sf_app), cr)
        nibbles.extend(decoded)
    nibbles = nibbles[:nibbles_needed]
    if len(nibbles) != nibbles_needed:
        raise LoRaDecodeError("Faltan nibbles para el payload y CRC.")

    bytes_wh = [nibbles[i] | (nibbles[i + 1] << 4)
                for i in range(0, nibbles_needed, 2)]
    for i, whitening_byte in enumerate(_whitening_sequence(length)):
        bytes_wh[i] ^= whitening_byte
    payload = bytes(bytes_wh[:length])
    crc_ok = None
    if has_crc:
        received_crc = bytes_wh[length] | (bytes_wh[length + 1] << 8)
        crc_ok = received_crc == _payload_crc(payload)

    observed = rows[:j + 2]
    pre_sfd_rows = [row for row in rows[:j] if row[1] is not None]
    pre_sfd_powers = [
        folded_power_at(true_start + row[0] * n_sym).astype(np.float32)
        for row in pre_sfd_rows
    ]
    folded_power = np.stack(pre_sfd_powers + header_powers + payload_powers)
    folded_section_starts = (
        len(pre_sfd_powers) - 2,
        len(pre_sfd_powers),
        len(pre_sfd_powers) + len(header_powers),
    )
    return LoRaFrame(
        payload=payload,
        crc_ok=crc_ok,
        header_checksum_ok=header_checksum_ok,
        sync_word=sync_word,
        coding_rate=cr,
        has_crc=has_crc,
        cfo_hz=float(cfo_hz),
        timing_correction_samples=true_start - coarse_start,
        drift_bins_per_symbol=float(drift),
        coarse_start_sample=coarse_start,
        frame_start_sample=true_start,
        sfd_start_sample=sfd_start,
        header_start_sample=header_start,
        payload_start_sample=payload_start,
        observed_bins=tuple(row[1] for row in observed),
        observed_offsets=tuple(row[0] for row in observed),
        header_bins=tuple(header_bins),
        payload_bins=tuple(payload_bins),
        header_symbols=tuple(header_symbols),
        payload_symbols=tuple(payload_symbols),
        header_concentrations=tuple(header_concentrations),
        payload_concentrations=tuple(payload_concentrations),
        folded_power=folded_power,
        folded_section_starts=folded_section_starts,
    )
