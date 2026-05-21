from abc import ABC, abstractmethod
import numpy as np

class DemoduladorBase(ABC):
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Identificador interno del demodulador."""
        pass

    @property
    @abstractmethod
    def nombre_mostrar(self) -> str:
        """Nombre visible en los menús de la interfaz gráfica."""
        pass

    @abstractmethod
    def configurar(self, sample_rate: float, fft_size: int):
        """
        Se llama al iniciar o cuando cambian los parámetros principales.
        Ideal para calcular los coeficientes FIR y limpiar buffers.
        """
        pass

    @abstractmethod
    def procesar(self, muestras_iq: np.ndarray) -> dict:
        """
        Recibe el chunk de muestras crudas y devuelve un diccionario con 
        todos los datos procesados listos para graficar o reproducir.
        """
        pass