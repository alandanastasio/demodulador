from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

class SDRBase(ABC):
    def __init__(self, rx_callback: Callable[[np.ndarray], None]):
        """
        rx_callback es la función a la que la radio le va a "escupir" 
        los chunks de datos IQ limpios (en formato np.complex128).
        """
        self.rx_callback = rx_callback
        self.is_running = False

    @property
    @abstractmethod
    def nombre(self) -> str:
        """Nombre del equipo, ej: 'HackRF One'"""
        pass

    @abstractmethod
    def configurar(self, sample_rate: float, center_freq: float):
        """Configura los parámetros iniciales antes de arrancar"""
        pass

    @abstractmethod
    def set_freq(self, freq_hz: float):
        pass

    @abstractmethod
    def set_sample_rate(self, sr_hz: float):
        pass

    @abstractmethod
    def set_gain(self, gain_db: int):
        pass

    @abstractmethod
    def start_rx(self):
        """Inicia la recepción de muestras"""
        pass

    @abstractmethod
    def stop_rx(self):
        """Detiene la recepción temporalmente"""
        pass

    @abstractmethod
    def close(self):
        """Apaga el equipo y libera el puerto USB (fundamental en Linux/libusb)"""
        pass