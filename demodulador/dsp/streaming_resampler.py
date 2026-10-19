"""Remuestreo polifásico continuo a partir de bloques de tamaño variable."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


class StreamingResampler:
    """Conserva los bordes FIR y entrega solo muestras con contexto completo.

    `resample_poly` centra su FIR y rellena con ceros el final de cada llamada.
    Retener una pequeña cola de entrada permite calcularla al llegar el bloque
    siguiente sin introducir artefactos en cada límite de bloque.
    """

    def __init__(self, up: int, down: int, window=('kaiser', 8.6)):
        if up < 1 or down < 1:
            raise ValueError('Los factores de remuestreo deben ser positivos.')
        common = gcd(up, down)
        self.up = up // common
        self.down = down // common
        self.window = window
        # scipy.signal.resample_poly usa 10 * max(up, down) taps por lado.
        self.radius = int(np.ceil(10 * max(self.up, self.down) / self.up)) + 1
        self.history_size = 4 * self.radius + self.down
        self.history = None
        self.history_start = 0
        self.total_input = 0
        self.next_output = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples)
        if len(samples) == 0:
            return samples[:0]
        if self.history is None:
            joined = samples
        else:
            joined = np.concatenate((self.history, samples), axis=0)

        result = resample_poly(joined, self.up, self.down,
                               axis=0, window=self.window)
        self.total_input += len(samples)
        output_start = self.history_start * self.up // self.down
        safe_next = max(0, (self.total_input - 1 - self.radius)
                        * self.up // self.down + 1)
        begin = self.next_output - output_start
        end = safe_next - output_start
        output = result[begin:end]
        self.next_output = safe_next

        # Alinear el inicio de la historia al período de la malla de salida.
        # Así el índice local de resample_poly coincide con el global.
        grid = self.down
        next_start = max(0, (self.total_input - self.history_size) // grid * grid)
        self.history = joined[next_start - self.history_start:].copy()
        self.history_start = next_start
        return output
