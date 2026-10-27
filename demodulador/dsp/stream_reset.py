"""Separa el estado DSP anterior y posterior a un corte del flujo IQ."""

from .demoduladores.btle import DemoduladorBTLE
from .demoduladores.lora import DemoduladorLoRa
from .demoduladores.lte_uplink import DemoduladorLTEUplink
from .demoduladores.wbfm_audio import DemoduladorWBFMAudio
from .demoduladores.wifi_ag import DemoduladorWiFiAG


def reset_demodulator_stream(demodulator, sample_rate=None, fft_size=None,
                             center_freq=None, clone_lora=False):
    """Conserva la configuración, pero descarta buffers y resultados en vuelo.

    Los demoduladores con workers reciben una instancia nueva: un worker viejo
    puede terminar, pero ya no puede publicar resultados en el objeto activo.
    LoRa dispone de su propia barrera de generación para esos workers.
    """
    if demodulator is None:
        return None
    sr = demodulator.sample_rate if sample_rate is None else sample_rate
    fs = demodulator.fft_size if fft_size is None else fft_size

    if isinstance(demodulator, DemoduladorLoRa) and not clone_lora:
        if sample_rate is not None or fft_size is not None:
            demodulator.configurar(sr, fs, demodulator.bandwidth_hz, demodulator.sf)
        else:
            demodulator.reset_stream()
        return demodulator

    if isinstance(demodulator, DemoduladorWBFMAudio):
        demodulator.procesar(None)

    fresh = type(demodulator)()
    args = (sr, fs)
    if isinstance(demodulator, DemoduladorLoRa):
        fresh.configurar(*args, demodulator.bandwidth_hz, demodulator.sf)
    elif isinstance(demodulator, DemoduladorBTLE):
        fresh.configurar(*args, bw_mhz=demodulator.bw_mhz,
                         decode_legacy=demodulator.decode_legacy)
        # El índice del canal determina la secuencia de whitening. Si se
        # pierde al reiniciar el flujo, se ven ráfagas pero ningún CRC válido.
        if center_freq is not None:
            fresh.set_center_frequency(center_freq)
        else:
            fresh.advertising_channel = demodulator.advertising_channel
    elif isinstance(demodulator, DemoduladorLTEUplink):
        fresh.configurar(*args, rb_count=demodulator.rb_count)
        # El barrido DL→UL sintoniza otra frecuencia a mitad de la sesión.
        # Sus muestras se descartan, pero conserva el Cell ID detectado.
        fresh.estado = demodulator.estado
        fresh.cell_id_guardada = demodulator.cell_id_guardada
        fresh.n_id_1 = demodulator.n_id_1
        fresh.n_id_2 = demodulator.n_id_2
    elif isinstance(demodulator, DemoduladorWiFiAG):
        fresh.configurar(*args, remove_dc=demodulator.remove_dc)
    else:
        fresh.configurar(*args)

    if hasattr(demodulator, 'window_type'):
        fresh.set_window(demodulator.window_type)
    if hasattr(demodulator, 'skip_metrics'):
        fresh.skip_metrics = demodulator.skip_metrics
    if isinstance(demodulator, DemoduladorWBFMAudio) and demodulator.result_callback is not None:
        fresh.result_callback = demodulator.result_callback
    return fresh
