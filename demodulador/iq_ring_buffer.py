import numpy as np

class IQRingBuffer:
    def __init__(self, max_samples):
        self.max_samples = int(max_samples)
        self.buffer = np.zeros(self.max_samples, dtype=np.complex64)
        self.write_idx = 0
        self.is_full = False
        
    def resize(self, new_max_samples):
        new_max_samples = int(new_max_samples)
        if new_max_samples == self.max_samples:
            return
            
        new_buffer = np.zeros(new_max_samples, dtype=np.complex64)
        
        # Copy existing data if any
        current_data = self.get_samples()
        keep = min(len(current_data), new_max_samples)
        
        if keep > 0:
            new_buffer[:keep] = current_data[-keep:]
            self.write_idx = keep % new_max_samples
            self.is_full = (keep == new_max_samples)
        else:
            self.write_idx = 0
            self.is_full = False
            
        self.max_samples = new_max_samples
        self.buffer = new_buffer
        
    def clear(self):
        self.write_idx = 0
        self.is_full = False
        
    def append(self, chunk):
        L = len(chunk)
        if L == 0: return
        
        if L >= self.max_samples:
            # Chunk is bigger than buffer, just keep the newest part
            self.buffer[:] = chunk[-self.max_samples:]
            self.write_idx = 0
            self.is_full = True
            return
            
        end_idx = self.write_idx + L
        if end_idx <= self.max_samples:
            self.buffer[self.write_idx:end_idx] = chunk
            self.write_idx = end_idx
            if self.write_idx == self.max_samples:
                self.write_idx = 0
                self.is_full = True
        else:
            # Wrap around
            overflow = end_idx - self.max_samples
            self.buffer[self.write_idx:self.max_samples] = chunk[:-overflow]
            self.buffer[0:overflow] = chunk[-overflow:]
            self.write_idx = overflow
            self.is_full = True

    def get_samples(self):
        if not self.is_full:
            return self.buffer[:self.write_idx]
        else:
            return np.concatenate((self.buffer[self.write_idx:], self.buffer[:self.write_idx]))
