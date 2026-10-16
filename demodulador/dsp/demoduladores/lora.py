"""Adaptador de recepción continua para el decodificador LoRa offline."""

from collections import deque
import threading
import time

import numpy as np

from .base import DemoduladorBase
from .lora_core import (
    LoRaConfig, LoRaDecodeError, LoRaIncompleteFrame,
    decode_capture,
)
from .sa import SpectrumAnalyzer


class DemoduladorLoRa(DemoduladorBase):
    def __init__(self):
        self.sample_rate = 2_000_000
        self.fft_size = 4096
        self.bandwidth_hz = 125_000
        self.sf = 7
        self._config = LoRaConfig(self.bandwidth_hz, self.sf)
        self._spectrum = SpectrumAnalyzer()
        self._lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._chunks = deque()
        self._buffer_samples = 0
        self._total_samples = 0
        self._processing = False
        self._last_attempt = 0.0
        self._next_search_abs = 0
        self._needed_end_abs = None
        self._preserve_from_abs = None
        self._pending_frames = deque(maxlen=4)
        self.last_frame = None

    @property
    def id(self):
        return "lora"

    @property
    def nombre_mostrar(self):
        return "LoRa"

    def configurar(self, sample_rate: float, fft_size: int,
                  bandwidth_hz: int | None = None, sf: int | None = None):
        bandwidth_hz = self.bandwidth_hz if bandwidth_hz is None else int(bandwidth_hz)
        sf = self.sf if sf is None else int(sf)
        config = LoRaConfig(bandwidth_hz, sf, int(sample_rate))
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.bandwidth_hz = bandwidth_hz
        self.sf = sf
        self._config = config
        self._spectrum.configurar(self.sample_rate, self.fft_size)
        self.reset_stream()

    def reset_stream(self):
        """Descarta IQ y resultados anteriores tras retuneo o cambio de modo."""
        with self._lock:
            self._generation += 1
            self._chunks.clear()
            self._buffer_samples = 0
            self._total_samples = 0
            self._processing = False
            self._last_attempt = 0.0
            self._next_search_abs = 0
            self._needed_end_abs = None
            self._preserve_from_abs = None
            self._pending_frames.clear()
            self.last_frame = None

    def close(self):
        with self._lock:
            self._closed = True
        self.reset_stream()

    def _search_window_samples(self):
        # SF12/BW125: 64 símbolos son ~2,1 s, tiempo suficiente para
        # conservar el preámbulo durante los reintentos de decodificación.
        return max(self.sample_rate, 64 * self._config.samples_per_symbol)

    def _trim_buffer_locked(self):
        search_window = self._search_window_samples()
        # Un segundo extra permite que llegue IQ nuevo mientras el worker
        # procesa una instantánea sin expulsar todavía el preámbulo.
        base_limit = max(int(self.sample_rate * 2), search_window + self.sample_rate)
        hard_limit = int(self.sample_rate * 16)
        limit = base_limit
        if self._preserve_from_abs is not None:
            limit = max(limit, self._total_samples - self._preserve_from_abs)
        limit = min(limit, hard_limit)
        while self._buffer_samples > limit:
            excess = self._buffer_samples - limit
            first = self._chunks[0]
            if len(first) <= excess:
                self._chunks.popleft()
                self._buffer_samples -= len(first)
            else:
                self._chunks[0] = first[excess:].copy()
                self._buffer_samples -= excess

    def _snapshot_chunks_locked(self, start_abs):
        """Entrega al worker sólo el IQ desde el inicio elegido para la búsqueda."""
        skip = start_abs - (self._total_samples - self._buffer_samples)
        parts = []
        for chunk in self._chunks:
            if skip >= len(chunk):
                skip -= len(chunk)
                continue
            parts.append(chunk[skip:])
            skip = 0
        return tuple(parts)

    @staticmethod
    def _packet_waterfall(samples, sample_rate):
        """FFT del IQ recibido, quitando sólo el offset DC del paquete."""
        fft_size = 1024
        if len(samples) < fft_size:
            raise ValueError("La trama es demasiado corta para su waterfall.")
        count = min(4096, 1 + (len(samples) - fft_size) // 256)
        starts = np.linspace(0, len(samples) - fft_size, count, dtype=np.int64)
        frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / sample_rate))
        power_db = np.empty((fft_size, count), dtype=np.float32)
        window = np.hanning(fft_size).astype(np.float32)
        offsets = np.arange(fft_size)
        dc = np.mean(samples, dtype=np.complex128)
        for begin in range(0, count, 128):
            end = min(begin + 128, count)
            frames = samples[starts[begin:end, None] + offsets] - dc
            spectrum = np.fft.fftshift(
                np.fft.fft(frames * window, axis=1), axes=1
            )
            power_db[:, begin:end] = (
                20 * np.log10(np.maximum(np.abs(spectrum), 1e-8))
            ).T
        peak_db = float(np.percentile(power_db, 99.7))
        return {
            'power_db': power_db,
            'freq_hz': frequencies,
            'duration_ms': len(samples) * 1000.0 / sample_rate,
            'levels_db': (peak_db - 50, peak_db),
        }

    def _decode_worker(self, chunks, snapshot_start_abs, config, generation):
        frame = None
        frame_waterfall = None
        incomplete = None
        try:
            snapshot = np.concatenate(chunks)
            frame = decode_capture(snapshot, config)
            visual_end = (
                frame.payload_start_sample
                + len(frame.payload_symbols) * config.samples_per_symbol
            )
            frame_waterfall = self._packet_waterfall(
                snapshot[frame.frame_start_sample:visual_end], config.sample_rate
            )
        except LoRaIncompleteFrame as exc:
            incomplete = exc
        except LoRaDecodeError:
            pass
        except Exception as exc:
            print(f"Error en demodulador LoRa: {exc}")
        finally:
            with self._lock:
                if generation != self._generation or self._closed:
                    return
                if frame is not None:
                    self._pending_frames.append((frame, frame_waterfall))
                    self._next_search_abs = (
                        snapshot_start_abs + frame.payload_start_sample
                        + len(frame.payload_symbols) * config.samples_per_symbol
                    )
                    self._needed_end_abs = None
                    self._preserve_from_abs = None
                elif incomplete is not None:
                    self._needed_end_abs = snapshot_start_abs + incomplete.required_samples
                    self._preserve_from_abs = max(
                        0, snapshot_start_abs + incomplete.frame_start_sample
                        - 4 * config.samples_per_symbol
                    )
                else:
                    self._needed_end_abs = None
                    self._preserve_from_abs = None
                self._processing = False
                self._trim_buffer_locked()

    def procesar(self, muestras_iq: np.ndarray) -> dict | None:
        if muestras_iq is None:
            self.reset_stream()
            return None

        spectrum = self._spectrum.procesar(muestras_iq)
        samples = np.asarray(muestras_iq)
        worker_args = None
        if len(samples):
            chunk = samples.astype(np.complex64, copy=True)
            with self._lock:
                if not self._closed:
                    self._chunks.append(chunk)
                    self._buffer_samples += len(chunk)
                    self._total_samples += len(chunk)
                    self._trim_buffer_locked()

                    available = self._total_samples - max(
                        self._next_search_abs,
                        self._total_samples - self._buffer_samples
                    )
                    min_samples = max(
                        12 * self._config.samples_per_symbol,
                        int(self.sample_rate * 0.25)
                    )
                    now = time.monotonic()
                    waiting_for_frame = self._needed_end_abs is not None
                    ready = not waiting_for_frame or self._total_samples >= self._needed_end_abs
                    retry_due = waiting_for_frame and ready
                    if (not self._processing and ready and available >= min_samples
                            and (retry_due or now - self._last_attempt >= 0.25)):
                        buffer_start_abs = self._total_samples - self._buffer_samples
                        search_start_abs = max(buffer_start_abs, self._next_search_abs)
                        if not waiting_for_frame:
                            # La ventana debe cubrir también las tramas lentas:
                            # con SF12/BW125 un segundo puede empezar ya
                            # después del preámbulo.
                            search_start_abs = max(
                                search_start_abs,
                                self._total_samples - self._search_window_samples(),
                            )
                        self._processing = True
                        self._last_attempt = now
                        worker_args = (
                            self._snapshot_chunks_locked(search_start_abs),
                            search_start_abs,
                            self._config,
                            self._generation,
                        )

        if worker_args is not None:
            threading.Thread(
                target=self._decode_worker, args=worker_args, daemon=True
            ).start()

        if spectrum is not None:
            metrics = {}
            with self._lock:
                if self._pending_frames:
                    self.last_frame, frame_waterfall = self._pending_frames.popleft()
                    metrics['lora_frame'] = self.last_frame
                    metrics['lora_waterfall'] = frame_waterfall
            spectrum['metricas'] = metrics
        return spectrum
