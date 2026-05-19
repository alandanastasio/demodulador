import numpy as np
from scipy.signal import resample_poly
import math

def sdr_a_osciloscopio(archivo_npz, archivo_csv, fc_jupyter=10e6, limite_muestras=None):
    print(f"Cargando archivo SDR: {archivo_npz}...")
    data = np.load(archivo_npz)
    
    # Tu PyQt guarda estos datos automáticamente en el npz
    iq_sdr = data['raw_iq']
    fs_sdr = float(data['sample_rate'])
    
    # Si la grabación es muy larga, limitamos a 1 segundo para no saturar la RAM
    # (Interpolar a 40 MHz consume bastante memoria)
    if limite_muestras and len(iq_sdr) > limite_muestras:
        print(f"Recortando a {limite_muestras} muestras para el análisis...")
        iq_sdr = iq_sdr[:limite_muestras]

    # 1. RE-MUESTREO (Oversampling)
    # El Jupyter espera una señal en 10 MHz. Por Teorema de Nyquist, 
    # necesitamos una frecuencia de muestreo de más del doble. Usamos 40 MHz.
    fs_nueva = 40e6 
    
    print(f"1. Interpolando velocidad de {fs_sdr/1e6} MHz a {fs_nueva/1e6} MHz...")
    gcd = math.gcd(int(fs_nueva), int(fs_sdr))
    up = int(fs_nueva // gcd)
    down = int(fs_sdr // gcd)
    
    iq_rapida = resample_poly(iq_sdr, up, down)
    
    # 2. DIGITAL UPCONVERSION (DUC)
    print("2. Generando portadora analítica y modulando señal a 10 MHz...")
    t = np.arange(len(iq_rapida)) / fs_nueva
    
    # Fabricamos la portadora artificial a 10 MHz y multiplicamos
    portadora = np.exp(1j * 2 * np.pi * fc_jupyter * t)
    señal_modulada = iq_rapida * portadora
    
    # 3. EXTRACCIÓN DE TENSIÓN FÍSICA
    # Como la señal ya no está en 0 Hz sino en 10 MHz, 
    # ahora SÍ es seguro borrar la parte imaginaria sin romper el FM.
    print("3. Extrayendo tensión real (simulando hardware)...")
    v = np.real(señal_modulada)
    
    # 4. EXPORTACIÓN A FORMATO OSCILOSCOPIO
    print(f"4. Exportando CSV en formato Siglent a {archivo_csv}...")
    datos_exportar = np.column_stack((t, v))
    encabezado = "Generado por DUC Converter\nTime(s),Voltage(V)"
    
    np.savetxt(
        archivo_csv, 
        datos_exportar, 
        delimiter=",", 
        header=encabezado, 
        comments='', 
        fmt=['%.8e', '%.8e']
    )
    print("¡Proceso terminado con éxito!")

# ==========================================
# CONFIGURACIÓN
# ==========================================
ARCHIVO_NPZ = "muestras_iq_20260515_163254.npz" # Poné el nombre de tu archivo grabado
ARCHIVO_CSV = "prueba.CSV"

# Procesamos 1 segundo de grabación (ej: 2.4 millones de muestras si grabaste a 2.4 MHz)
# Esto es suficiente para el Jupyter y evita que tu compu se quede sin RAM.
sdr_a_osciloscopio(ARCHIVO_NPZ, ARCHIVO_CSV, fc_jupyter=10e6, limite_muestras=2400000)