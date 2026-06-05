import numpy as np

class TraceManager:
    def __init__(self):
        self.trace_mode = "White clear"
        self.max_hold_data = None
        self.avg_buffer = None
        self.avg_index = 0
        self.avg_count = 0
        self.AVG_MAX = 100

    def set_mode(self, mode_text):
        """Cambia el modo de la traza y resetea los buffers"""
        self.trace_mode = mode_text
        self.reset()

    def reset(self):
        """Limpia la memoria de las trazas (útil al cambiar tamaño de FFT)"""
        self.max_hold_data = None
        self.avg_buffer = None
        self.avg_index = 0
        self.avg_count = 0

    def process(self, psd_in):
        """Recibe el espectro en vivo y devuelve el espectro procesado"""
        
        if self.trace_mode == "White clear":
            return psd_in
            
        elif self.trace_mode == "Max Hold":
            if self.max_hold_data is None or len(self.max_hold_data) != len(psd_in):
                self.max_hold_data = psd_in.copy()
            else:
                # Compara punto por punto y se queda con el más alto
                self.max_hold_data = np.maximum(self.max_hold_data, psd_in) 
            return self.max_hold_data

        elif self.trace_mode == "Average":
            if self.avg_buffer is None or self.avg_buffer.shape[1] != len(psd_in):
                # Matriz de 100 filas x N frecuencias
                self.avg_buffer = np.zeros((self.AVG_MAX, len(psd_in)))
                self.avg_index = 0
                self.avg_count = 0
            
            # Guarda la muestra actual y avanza cíclicamente
            self.avg_buffer[self.avg_index] = psd_in
            self.avg_index = (self.avg_index + 1) % self.AVG_MAX
            if self.avg_count < self.AVG_MAX:
                self.avg_count += 1
            
            # Devuelve el promedio
            return np.mean(self.avg_buffer[:self.avg_count], axis=0)
            
        return psd_in