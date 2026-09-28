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

    def _trim_buffer_locked(self):
        base_limit = int(self.sample_rate * 2)
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

    def _recent_samples_locked(self, count):
        """Copia sólo la ventana que necesitan los gráficos, no todo el buffer DSP."""
        parts = []
        remaining = count
        for chunk in reversed(self._chunks):
            if remaining <= 0:
                break
            parts.append(chunk[-remaining:])
            remaining -= len(parts[-1])
        return np.concatenate(parts[::-1]) if parts else np.empty(0, np.complex64)

    @staticmethod
    def _visual_window_samples(config):
        return min(
            262_144,
            max(int(config.sample_rate * 0.012), 3 * config.samples_per_symbol),
        )

    @staticmethod
    def _visual_data(samples, sample_rate, complete_frame=False):
        if len(samples) < 2:
            return None
        # Muestreamos pares contiguos en distintas posiciones de la trama.
        # Así el costo queda acotado aunque la trama dure varios segundos, y
        # la frecuencia instantánea no sufre aliasing por el salto del gráfico.
        smooth = min(16, len(samples) - 1)
        count = min(6000 if complete_frame else 3000, len(samples) - smooth)
        indices = np.linspace(0, len(samples) - smooth - 1, count, dtype=np.int64)
        offsets = np.arange(smooth)
        positions = indices[:, None] + offsets
        cross = samples[positions + 1] * np.conjugate(samples[positions])
        average_cross = cross.mean(axis=1)
        magnitude = np.abs(samples[positions]).mean(axis=1)
        return {
            'time_ms': (indices + smooth / 2) * (1000.0 / sample_rate),
            'magnitude': magnitude,
            'freq_khz': np.angle(average_cross) * (sample_rate / (2000.0 * np.pi)),
            'duration_ms': len(samples) * 1000.0 / sample_rate,
            'complete_frame': complete_frame,
        }

    def _decode_worker(self, chunks, snapshot_start_abs, search_start_abs,
                       config, generation):
        frame = None
        frame_visual = None
        incomplete = None
        try:
            snapshot = np.concatenate(chunks)
            search_start = max(0, search_start_abs - snapshot_start_abs)
            if search_start < len(snapshot):
                frame = decode_capture(snapshot, config, start_sample=search_start)
                visual_end = (
                    frame.payload_start_sample
                    + len(frame.payload_symbols) * config.samples_per_symbol
                )
                frame_visual = self._visual_data(
                    snapshot[frame.frame_start_sample:visual_end],
                    config.sample_rate, complete_frame=True,
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
                    self._pending_frames.append((frame, frame_visual))
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
        visual_samples = None
        if len(samples):
            chunk = samples.astype(np.complex64, copy=True)
            with self._lock:
                if not self._closed:
                    self._chunks.append(chunk)
                    self._buffer_samples += len(chunk)
                    self._total_samples += len(chunk)
                    self._trim_buffer_locked()

                    if spectrum is not None and self.last_frame is None:
                        visual_samples = self._recent_samples_locked(
                            self._visual_window_samples(self._config)
                        )

                    available = self._total_samples - max(
                        self._next_search_abs,
                        self._total_samples - self._buffer_samples
                    )
                    min_samples = max(
                        12 * self._config.samples_per_symbol,
                        int(self.sample_rate * 0.25)
                    )
                    now = time.monotonic()
                    ready = self._needed_end_abs is None or self._total_samples >= self._needed_end_abs
                    if (not self._processing and ready and available >= min_samples
                            and now - self._last_attempt >= 0.25):
                        self._processing = True
                        self._last_attempt = now
                        worker_args = (
                            tuple(self._chunks),
                            self._total_samples - self._buffer_samples,
                            self._next_search_abs,
                            self._config,
                            self._generation,
                        )

        if worker_args is not None:
            threading.Thread(
                target=self._decode_worker, args=worker_args, daemon=True
            ).start()

        if spectrum is not None:
            metrics = {}
            if visual_samples is not None:
                metrics['lora_visual'] = self._visual_data(visual_samples, self.sample_rate)
            with self._lock:
                if self._pending_frames:
                    self.last_frame, frame_visual = self._pending_frames.popleft()
                    metrics['lora_frame'] = self.last_frame
                    metrics['lora_visual'] = frame_visual
            spectrum['metricas'] = metrics
        return spectrum
