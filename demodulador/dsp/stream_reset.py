"""Separa el estado DSP anterior y posterior a un corte del flujo IQ."""

from .demoduladores.btle import DemoduladorBTLE
from .demoduladores.lora import DemoduladorLoRa
from .demoduladores.lte_uplink import DemoduladorLTEUplink


def reset_demodulator_stream(demodulator):
    """Conserva la configuración, pero descarta buffers y resultados en vuelo.

    Los demoduladores con workers reciben una instancia nueva: un worker viejo
    puede terminar, pero ya no puede publicar resultados en el objeto activo.
    LoRa dispone de su propia barrera de generación para esos workers.
    """
    if isinstance(demodulator, DemoduladorLoRa):
        demodulator.reset_stream()
        return demodulator

    fresh = type(demodulator)()
    args = (demodulator.sample_rate, demodulator.fft_size)
    if isinstance(demodulator, DemoduladorBTLE):
        fresh.configurar(*args, bw_mhz=demodulator.bw_mhz)
    elif isinstance(demodulator, DemoduladorLTEUplink):
        fresh.configurar(*args, rb_count=demodulator.rb_count)
        # El barrido DL→UL sintoniza otra frecuencia a mitad de la sesión.
        # Sus muestras se descartan, pero conserva el Cell ID detectado.
        fresh.estado = demodulator.estado
        fresh.cell_id_guardada = demodulator.cell_id_guardada
        fresh.n_id_1 = demodulator.n_id_1
        fresh.n_id_2 = demodulator.n_id_2
    else:
        fresh.configurar(*args)

    if hasattr(demodulator, 'window_type'):
        fresh.set_window(demodulator.window_type)
    if hasattr(demodulator, 'skip_metrics'):
        fresh.skip_metrics = demodulator.skip_metrics
    return fresh
