import numpy as np
from dsp.demoduladores.wifi_ag import DemoduladorWiFiAG
import traceback

demod = DemoduladorWiFiAG()
demod.configurar(20e6, 4096)

# Load data
x = np.load('captura_bladerf.npy')
print("Loaded samples:", len(x))

demod._procesar_fondo(x)
if not demod.nuevos_datos_listos:
    print("No new data ready, maybe it didn't find a burst or it crashed?")
else:
    res = demod.last_heavy_results
    print("Success! evm_data keys:", res.get('evm_data', {}).keys() if res.get('evm_data') else None)
    
