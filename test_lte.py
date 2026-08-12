import numpy as np

fs = 256
pss_freq = np.zeros(fs, dtype=complex)
pss_freq[10:72] = np.exp(1j * np.random.uniform(0, 2*np.pi, 62)) # Mock PSS
pss_t = np.fft.ifft(pss_freq) * np.sqrt(fs)

chunk = np.zeros(2000, dtype=complex)
chunk[500:500+fs] = pss_t

# Add 6 subcarrier offset
icfo_hz = 6 * 15000.0
ts = 1.0 / 3.84e6
t_vector = np.arange(2000) * ts
chunk *= np.exp(1j * 2 * np.pi * icfo_hz * t_vector)

# Simulate extraction at 390
simbolo_pss = chunk[390:390+fs]
simbolo_f = np.fft.fftshift(np.fft.fft(simbolo_pss))

pss_rx = simbolo_f[10+6:72+6] # roughly idx_extr for offset 6
corr = np.abs(np.vdot(pss_freq[10:72], pss_rx))
print("Grid corr at pos 390:", corr)
print("Threshold:", 4.0 * np.sqrt(62 * fs))
