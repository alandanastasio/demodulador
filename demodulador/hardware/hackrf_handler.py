"""Recepción IQ de HackRF One para todos los modos de demodulación."""

import logging
import math
import threading

import numpy as np
from python_hackrf import pyhackrf

from .sdr_base import SDRBase


class HackRFHandler(SDRBase):
    MIN_SAMPLE_RATE = 2_000_000
    MAX_SAMPLE_RATE = 20_000_000
    MIN_FREQUENCY = 1_000_000
    MAX_FREQUENCY = 6_000_000_000

    def __init__(self, rx_callback):
        super().__init__(rx_callback)
        self.sdr = None
        self._lock = threading.RLock()
        self._closed = False
        self._has_started = False
        self._sample_rate = None
        self._baseband_bandwidth = None
        self._center_freq = None
        pyhackrf.pyhackrf_init()
        try:
            self.sdr = pyhackrf.pyhackrf_open()
            if self.sdr is None:
                raise RuntimeError("No se pudo abrir la HackRF.")
            self.sdr.set_rx_callback(self._internal_rx_callback)
        except Exception:
            if self.sdr is not None:
                try:
                    self.sdr.pyhackrf_close()
                except Exception:
                    logging.exception("No se pudo cerrar la HackRF tras fallar la apertura")
            pyhackrf.pyhackrf_exit()
            raise

    @property
    def nombre(self):
        return "HackRF One"

    def _require_open(self):
        if self._closed or self.sdr is None:
            raise RuntimeError("La HackRF está cerrada.")

    def validate_sample_rate(self, sr_hz: float):
        try:
            valid = math.isfinite(sr_hz) and self.MIN_SAMPLE_RATE <= sr_hz <= self.MAX_SAMPLE_RATE
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("La HackRF One admite tasas de 2 a 20 Msps.")

    def validate_frequency(self, freq_hz: float):
        try:
            valid = math.isfinite(freq_hz) and self.MIN_FREQUENCY <= freq_hz <= self.MAX_FREQUENCY
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("La frecuencia de HackRF debe estar entre 1 MHz y 6 GHz.")

    def _notify_discontinuity(self):
        if self.rx_callback is not None:
            self.rx_callback(None)

    def _actual_streaming_state(self, fallback):
        try:
            return bool(self.sdr.pyhackrf_is_streaming())
        except Exception:
            return fallback

    def _start_rx_native(self):
        # La biblioteca puede invocar el callback antes de que start_rx retorne.
        self.is_running = True
        try:
            self.sdr.pyhackrf_start_rx()
        except Exception:
            self.is_running = self._actual_streaming_state(False)
            raise
        self._has_started = True

    def _stop_rx_native(self):
        self.is_running = False
        try:
            self.sdr.pyhackrf_stop_rx()
        except Exception:
            self.is_running = self._actual_streaming_state(True)
            raise

    def _reconfigure(self, apply, restore):
        was_running = self.is_running or self._actual_streaming_state(False)
        if was_running:
            self._stop_rx_native()
        try:
            apply()
        except Exception:
            try:
                restore()
            except Exception as rollback_error:
                raise RuntimeError(
                    "Falló la configuración de HackRF y no se pudo restaurar la anterior."
                ) from rollback_error
            if was_running:
                self._notify_discontinuity()
                self._start_rx_native()
            raise
        if was_running:
            self._notify_discontinuity()
            self._start_rx_native()

    def _internal_rx_callback(self, device, buffer, buffer_length, valid_length):
        try:
            if not self.is_running:
                return 0
            if valid_length < 0 or valid_length > buffer_length or valid_length % 2:
                raise ValueError("Bloque IQ inválido recibido de HackRF.")
            raw_data = np.array(buffer[:valid_length], dtype=np.int8)
            samples = (raw_data[0::2] + 1j * raw_data[1::2]) / 128.0
            if self.rx_callback is not None:
                self.rx_callback(samples)
        except Exception:
            logging.exception("Error en el callback de recepción HackRF")
            self.is_running = False
            return -1
        return 0

    def configurar(self, sample_rate: float, center_freq: float):
        self.set_sample_rate(sample_rate)
        self.set_freq(center_freq)
        self.set_gain(8)
        self.set_vga_gain(16)

    def set_freq(self, freq_hz: float):
        self.validate_frequency(freq_hz)
        freq_hz = int(round(freq_hz))
        with self._lock:
            self._require_open()
            if freq_hz == self._center_freq:
                return
            previous = self._center_freq

            def apply():
                self.sdr.pyhackrf_set_freq(freq_hz)
                self._center_freq = freq_hz

            def restore():
                if previous is not None:
                    self.sdr.pyhackrf_set_freq(previous)

            self._reconfigure(apply, restore)

    def set_sample_rate(self, sr_hz: float):
        self.validate_sample_rate(sr_hz)
        sr_hz = int(round(sr_hz))
        with self._lock:
            self._require_open()
            if sr_hz == self._sample_rate:
                return
            bandwidth = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(
                int(sr_hz * 0.75)
            )
            previous_rate = self._sample_rate
            previous_bandwidth = self._baseband_bandwidth

            def apply():
                self.sdr.pyhackrf_set_sample_rate(sr_hz)
                self.sdr.pyhackrf_set_baseband_filter_bandwidth(bandwidth)
                self._sample_rate = sr_hz
                self._baseband_bandwidth = bandwidth

            def restore():
                if previous_rate is not None:
                    self.sdr.pyhackrf_set_sample_rate(previous_rate)
                    self.sdr.pyhackrf_set_baseband_filter_bandwidth(previous_bandwidth)

            self._reconfigure(apply, restore)

    def set_gain(self, gain_db: int):
        if not isinstance(gain_db, int) or gain_db not in range(0, 41, 8):
            raise ValueError("La ganancia LNA de HackRF debe ser 0–40 dB en pasos de 8 dB.")
        with self._lock:
            self._require_open()
            self.sdr.pyhackrf_set_lna_gain(gain_db)

    def set_vga_gain(self, gain_db: int):
        if not isinstance(gain_db, int) or gain_db not in range(0, 63, 2):
            raise ValueError("La ganancia VGA de HackRF debe ser 0–62 dB en pasos de 2 dB.")
        with self._lock:
            self._require_open()
            self.sdr.pyhackrf_set_vga_gain(gain_db)

    def start_rx(self):
        with self._lock:
            self._require_open()
            if self._actual_streaming_state(self.is_running):
                if self.is_running:
                    return
                self._stop_rx_native()
            self.is_running = False
            if self._has_started:
                self._notify_discontinuity()
            self._start_rx_native()

    def stop_rx(self):
        with self._lock:
            if self._closed:
                return
            if not self.is_running and not self._actual_streaming_state(False):
                return
            self._stop_rx_native()
            self._notify_discontinuity()

    def close(self):
        with self._lock:
            if self._closed:
                return
            errors = []
            if self.is_running or self._actual_streaming_state(False):
                try:
                    self._stop_rx_native()
                except Exception as exc:
                    errors.append(exc)
            try:
                self.sdr.pyhackrf_close()
            except Exception as exc:
                errors.append(exc)
            try:
                pyhackrf.pyhackrf_exit()
            except Exception as exc:
                errors.append(exc)
            self.sdr = None
            self.is_running = False
            self._closed = True
            if errors:
                raise RuntimeError("No se pudo cerrar correctamente la HackRF.") from errors[0]
