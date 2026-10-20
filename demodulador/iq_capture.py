"""Grabación secuencial de IQ y lectura de capturas verificadas."""

from collections import deque
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import Condition, Thread

import numpy as np


class CaptureError(RuntimeError):
    pass


def _valid_frequency(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise CaptureError(f"{name} debe ser un número positivo y finito.")
    return value


class IQRecorder:
    """Guarda cada bloque entregado por el SDR, en orden y sin convertir su dtype.

    La cola tiene un límite explícito. Si el disco no alcanza el ritmo de la
    recepción, la captura se invalida en lugar de publicar un archivo truncado.
    """

    def __init__(self, path, center_freq, sample_rate, on_done=None,
                 max_pending_bytes=256 * 1024 * 1024):
        self.path = Path(path)
        if self.path.suffix.lower() != ".iq":
            raise CaptureError("La captura debe tener extensión .iq.")
        self.metadata_path = self.path.with_suffix(".json")
        self._temp_path = Path(str(self.path) + ".part")
        self._temp_metadata_path = Path(str(self.metadata_path) + ".part")
        self.center_freq = _valid_frequency(center_freq, "center_freq")
        self.sample_rate = _valid_frequency(sample_rate, "sample_rate")
        self.max_pending_bytes = int(max_pending_bytes)
        if self.max_pending_bytes <= 0:
            raise ValueError("max_pending_bytes debe ser positivo")
        self.on_done = on_done
        self._condition = Condition()
        self._queue = deque()
        self._pending_bytes = 0
        self._closed = False
        self._error = None
        self._dtype = None
        self._sample_count = 0
        self._block_count = 0
        self._result = None
        if self.path.exists() or self.metadata_path.exists():
            raise CaptureError(f"La captura ya existe: {self.path}")
        self._file = self._temp_path.open("xb", buffering=1024 * 1024)
        self._thread = Thread(target=self._write_loop, daemon=True,
                              name="iq-recorder")
        self._thread.start()

    @property
    def error(self):
        with self._condition:
            return self._error

    @property
    def result(self):
        return self._result

    def append(self, samples):
        array = np.asarray(samples)
        if array.ndim != 1 or array.dtype.kind != "c" or not array.size:
            self.abort("Bloque IQ vacío o inválido.")
            raise CaptureError("Bloque IQ vacío o inválido.")
        with self._condition:
            if self._closed or self._error:
                raise CaptureError(self._error or "La grabación ya terminó.")
            if self._dtype is None:
                self._dtype = array.dtype
            elif array.dtype != self._dtype:
                self._fail_locked("Cambió el formato de las muestras IQ durante la grabación.")
                raise CaptureError(self._error)
            if self._pending_bytes + array.nbytes > self.max_pending_bytes:
                self._fail_locked("El disco no alcanzó el ritmo de captura: cola IQ llena.")
                raise CaptureError(self._error)
            # El SDR puede reutilizar su buffer al volver del callback.
            block = np.array(array, copy=True, order="C")
            self._queue.append(block)
            self._pending_bytes += block.nbytes
            self._condition.notify()

    def _fail_locked(self, reason):
        self._error = reason
        self._closed = True
        self._queue.clear()
        self._condition.notify_all()

    def abort(self, reason):
        with self._condition:
            if self._result is None and self._error is None:
                self._fail_locked(reason)

    def finish(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def wait(self):
        self._thread.join()
        return self._result

    def _write_loop(self):
        digest = sha256()
        failure = None
        try:
            with self._file as output:
                while True:
                    with self._condition:
                        while not self._queue and not self._closed:
                            self._condition.wait()
                        if self._error:
                            raise CaptureError(self._error)
                        if not self._queue:
                            break
                        block = self._queue.popleft()
                    raw = memoryview(block).cast("B")
                    offset = 0
                    while offset < len(raw):
                        written = output.write(raw[offset:])
                        if not written:
                            raise OSError("No se pudo completar la escritura IQ.")
                        offset += written
                    digest.update(raw)
                    self._sample_count += len(block)
                    self._block_count += 1
                    with self._condition:
                        self._pending_bytes -= block.nbytes
                if not self._sample_count:
                    raise CaptureError("La grabación no recibió muestras IQ.")
                output.flush()
                os.fsync(output.fileno())

            metadata = {
                "format": "demodulador-iq-v1",
                "dtype": self._dtype.str,
                "sample_count": self._sample_count,
                "block_count": self._block_count,
                "sample_rate": self.sample_rate,
                "center_freq": self.center_freq,
                "sha256": digest.hexdigest(),
            }
            with self._temp_metadata_path.open("x", encoding="utf-8") as output:
                json.dump(metadata, output, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            with self._condition:
                if self._error:
                    raise CaptureError(self._error)
            os.replace(self._temp_path, self.path)
            os.replace(self._temp_metadata_path, self.metadata_path)
            self._result = metadata
        except Exception as exc:
            failure = str(exc)
            with self._condition:
                self._error = failure
                self._closed = True
            self.path.unlink(missing_ok=True)
            self.metadata_path.unlink(missing_ok=True)
        finally:
            self._temp_path.unlink(missing_ok=True)
            self._temp_metadata_path.unlink(missing_ok=True)
            if self.on_done is not None:
                self.on_done(self, failure)


def load_capture(path, verify=True):
    """Devuelve (IQ, frecuencia central, tasa); soporta NPZ anteriores."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            samples = archive["raw_iq"]
            center_freq = _valid_frequency(archive["center_freq"], "center_freq")
            sample_rate = _valid_frequency(archive["sample_rate"], "sample_rate")
    elif path.suffix.lower() == ".iq":
        with path.with_suffix(".json").open(encoding="utf-8") as source:
            metadata = json.load(source)
        if metadata.get("format") != "demodulador-iq-v1":
            raise CaptureError("Formato de captura IQ desconocido.")
        dtype = np.dtype(metadata["dtype"])
        if dtype.kind != "c":
            raise CaptureError("La captura no contiene muestras complejas.")
        count = int(metadata["sample_count"])
        if count <= 0 or count * dtype.itemsize != path.stat().st_size:
            raise CaptureError("La captura IQ está incompleta o tiene muestras extra.")
        center_freq = _valid_frequency(metadata["center_freq"], "center_freq")
        sample_rate = _valid_frequency(metadata["sample_rate"], "sample_rate")
        if verify:
            digest = sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != metadata["sha256"]:
                raise CaptureError("La captura IQ no pasó la verificación de integridad.")
        samples = np.memmap(path, dtype=dtype, mode="r", shape=(count,))
    else:
        raise CaptureError("Seleccioná un archivo .iq o .npz.")
    if samples.ndim != 1 or samples.dtype.kind != "c" or not samples.size:
        raise CaptureError("La captura no contiene un vector IQ válido.")
    return samples, center_freq, sample_rate
