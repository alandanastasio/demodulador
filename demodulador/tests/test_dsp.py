import pytest
import numpy as np
from dsp.demoduladores.sa import SpectrumAnalyzer
from trace_manager import TraceManager

def test_spectrum_analyzer_detects_peak():
    """
    Inyecta una onda senoidal pura y verifica que el Analizador de Espectro
    (SpectrumAnalyzer) detecte la máxima potencia exactamente en ese bin.
    """
    sa = SpectrumAnalyzer()
    sample_rate = 2e6
    fft_size = 4096
    sa.configurar(sample_rate, fft_size)
    
    # Reseteamos el throttle de 30 FPS para que el test sea determinista
    sa.last_update = 0 
    
    # Generamos una señal: Un tono senoidal complejo (CW) desplazado +500 kHz del centro
    target_freq = 500e3 
    t = np.arange(fft_size) / sample_rate
    # x(t) = e^(j * 2 * pi * f * t)
    signal = np.exp(1j * 2 * np.pi * target_freq * t)
    
    # Procesamos la señal
    result = sa.procesar(signal)
    
    assert result is not None, "El SpectrumAnalyzer no devolvió resultados"
    assert 'psd_rf' in result, "El resultado debe contener el diccionario 'psd_rf'"
    
    psd = result['psd_rf']
    
    # Calculamos qué bin debería tener la energía máxima.
    # El arreglo PSD devuelto por sa.py ya tiene hecho un fftshift.
    # El bin central (DC) está en fft_size // 2.
    # Cada bin representa (sample_rate / fft_size) Hz.
    bin_width = sample_rate / fft_size
    expected_bin = int((fft_size // 2) + (target_freq / bin_width))
    
    actual_max_bin = np.argmax(psd)
    
    # Aceptamos un margen de +/- 1 bin por redondeos
    assert abs(actual_max_bin - expected_bin) <= 1, f"Pico detectado en bin {actual_max_bin}, se esperaba ~{expected_bin}"


def test_trace_manager_max_hold():
    """
    Verifica que el modo Max Hold conserve estrictamente el valor máximo histórico
    por cada bin de frecuencia.
    """
    tm = TraceManager()
    tm.set_mode("Max Hold")
    
    espectro_1 = np.array([-100.0, -90.0, -80.0])
    espectro_2 = np.array([-110.0, -85.0, -75.0])
    espectro_3 = np.array([-95.0,  -95.0, -85.0])
    
    res_1 = tm.process(espectro_1)
    np.testing.assert_array_equal(res_1, [-100.0, -90.0, -80.0])
    
    res_2 = tm.process(espectro_2)
    # Debe retener el máximo de espectro_1 y espectro_2
    np.testing.assert_array_equal(res_2, [-100.0, -85.0, -75.0])
    
    res_3 = tm.process(espectro_3)
    # Debe retener el máximo acumulado vs espectro_3
    np.testing.assert_array_equal(res_3, [-95.0, -85.0, -75.0])


def test_trace_manager_average():
    """
    Verifica que el modo Average calcule correctamente la media matemática
    (Promedio) de las señales ingresadas a lo largo del tiempo.
    """
    tm = TraceManager()
    tm.set_mode("Average")
    
    espectro_1 = np.array([-100.0, -50.0])
    espectro_2 = np.array([-80.0,  -50.0])
    espectro_3 = np.array([-60.0,  -80.0])
    
    res_1 = tm.process(espectro_1)
    np.testing.assert_array_almost_equal(res_1, [-100.0, -50.0])
    
    res_2 = tm.process(espectro_2)
    # Promedio matemático de (-100 + -80)/2 y (-50 + -50)/2
    np.testing.assert_array_almost_equal(res_2, [-90.0, -50.0])
    
    res_3 = tm.process(espectro_3)
    # Promedio matemático de (-100 + -80 + -60)/3 y (-50 + -50 + -80)/3
    np.testing.assert_array_almost_equal(res_3, [-80.0, -60.0])
