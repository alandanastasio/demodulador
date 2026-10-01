"""Un cambio de configuración no debe reutilizar IQ o resultados anteriores."""

import numpy as np

from dsp.demoduladores.btle import DemoduladorBTLE
from dsp.demoduladores.lora import DemoduladorLoRa
from dsp.demoduladores.lte_uplink import DemoduladorLTEUplink
from dsp.demoduladores.sa import SpectrumAnalyzer
from dsp.demoduladores.wifi_ag import DemoduladorWiFiAG
from dsp.stream_reset import reset_demodulator_stream


def test_preserves_mode_parameters_and_discards_btle_samples():
    old = DemoduladorBTLE()
    old.configurar(20_000_000, 4096, bw_mhz=2)
    old.buffer = np.ones(100, np.complex64)

    fresh = reset_demodulator_stream(old)

    assert fresh is not old
    assert fresh.bw_mhz == 2
    assert fresh.sample_rate == 20_000_000
    assert len(fresh.buffer) == 0


def test_old_worker_results_cannot_reach_new_demodulator():
    old = DemoduladorWiFiAG()
    old.configurar(20_000_000, 4096)
    old.nuevos_datos_listos = True
    old.last_heavy_results = {'old': True}

    fresh = reset_demodulator_stream(old)
    old.last_heavy_results = {'finished_late': True}

    assert fresh is not old
    assert not fresh.nuevos_datos_listos
    assert fresh.last_heavy_results == {}


def test_preserves_uplink_allocation_and_spectrum_window():
    uplink = DemoduladorLTEUplink()
    uplink.configurar(7_680_000, 512, rb_count=25)
    uplink.skip_metrics = True
    uplink.estado = 'WAITING_UL'
    uplink.cell_id_guardada = 123
    fresh_uplink = reset_demodulator_stream(uplink)
    assert fresh_uplink.rb_count == 25
    assert fresh_uplink.skip_metrics is True
    assert fresh_uplink.estado == 'WAITING_UL'
    assert fresh_uplink.cell_id_guardada == 123

    spectrum = SpectrumAnalyzer()
    spectrum.configurar(8_000_000, 2048)
    spectrum.set_window('hanning')
    fresh_spectrum = reset_demodulator_stream(spectrum)
    assert fresh_spectrum.window_type == 'hanning'


def test_lora_uses_its_generation_barrier():
    lora = DemoduladorLoRa()
    lora.configurar(2_000_000, 4096, 125_000, 7)
    generation = lora._generation
    lora._chunks.append(np.ones(128, dtype=np.complex64))
    lora._buffer_samples = 128

    assert reset_demodulator_stream(lora) is lora
    assert lora._generation == generation + 1
    assert lora._buffer_samples == 0
    lora.close()
