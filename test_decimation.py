import numpy as np
import scipy.signal as signal
from demodulador.dsp.demoduladores.lte import generar_pss_time

pss_time = generar_pss_time(2048)
pss0 = pss_time[0]

# Create a dummy chunk of 307200 samples with noise
chunk = np.random.randn(307200) + 1j * np.random.randn(307200)
chunk *= 0.1

# Insert PSS at position 100000
chunk[100000:100000+2048] += pss0

# Normal correlation
corr = signal.correlate(chunk, pss0, mode='valid', method='fft')
print("Normal peak:", np.max(np.abs(corr)), "at", np.argmax(np.abs(corr)))

# Decimated correlation
step = 16
chunk_dec = chunk[::step]
pss_dec = pss0[::step]

corr_dec = signal.correlate(chunk_dec, pss_dec, mode='valid', method='fft')
peak_dec_pos = np.argmax(np.abs(corr_dec))
print("Decimated peak:", np.max(np.abs(corr_dec)), "at", peak_dec_pos, "corresponding to", peak_dec_pos * step)

# Local search
search_start = max(0, peak_dec_pos * step - step)
search_end = min(len(chunk), peak_dec_pos * step + step + 2048)
chunk_local = chunk[search_start:search_end]
corr_local = signal.correlate(chunk_local, pss0, mode='valid', method='direct')
local_peak = np.argmax(np.abs(corr_local))
final_pos = search_start + local_peak
print("Final pos:", final_pos)
