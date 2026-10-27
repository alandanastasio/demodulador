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
    old.set_center_frequency(2_426_000_000)
    old.buffer = np.ones(100, np.complex64)

    fresh = reset_demodulator_stream(old)

    assert fresh is not old
    assert fresh.bw_mhz == 2
    assert fresh.sample_rate == 20_000_000
    assert fresh.advertising_channel == 38
    assert len(fresh.buffer) == 0


def test_preserves_le_1m_phy_metrics_mode_after_stream_reset():
    old = DemoduladorBTLE()
    old.configurar(20_000_000, 4096, bw_mhz=1, decode_legacy=False)
    old.set_center_frequency(2_402_000_000)

    fresh = reset_demodulator_stream(old)

    assert fresh.bw_mhz == 1
    assert fresh.decode_legacy is False
    assert fresh.advertising_channel == 37


def test_old_worker_results_cannot_reach_new_demodulator():
    old = DemoduladorWiFiAG()
    old.configurar(20_000_000, 4096, remove_dc=True)
    old.nuevos_datos_listos = True
    old.last_heavy_results = {'old': True}

    fresh = reset_demodulator_stream(old)
    old.last_heavy_results = {'finished_late': True}

    assert fresh is not old
    assert fresh.remove_dc is True
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


def test_btle_procesar_none_resets_buffer_without_crashing():
    btle = DemoduladorBTLE()
    btle.configurar(20_000_000, 1024, bw_mhz=2)
    btle.buffer = np.ones(256, dtype=np.complex64)
    btle._recent_packet_starts = [(10, b"abc")]
    btle.last_burst_metrics = {"f1_avg_khz": 250.0}

    assert btle.procesar(None) is None
    assert len(btle.buffer) == 0
    assert btle._recent_packet_starts == []
    assert btle.last_burst_metrics is None


def test_wbfm_audio_preserves_result_callback_on_stream_reset():
    from dsp.demoduladores.wbfm_audio import DemoduladorWBFMAudio

    cb = lambda res: None
    wbfm = DemoduladorWBFMAudio()
    wbfm.configurar(2_400_000, 4096)
    wbfm.result_callback = cb

    fresh = reset_demodulator_stream(wbfm, sample_rate=2_000_000, fft_size=2048)
    assert fresh is not wbfm
    assert fresh.result_callback is cb
    assert fresh.sample_rate == 2_000_000
    assert fresh.fft_size == 2048


def test_iq_ring_buffer_resize_and_append():
    from iq_ring_buffer import IQRingBuffer

    buf = IQRingBuffer(max_samples=100)
    buf.append(np.arange(60, dtype=np.complex64))
    assert len(buf.get_samples()) == 60

    buf.resize(40)
    assert buf.max_samples == 40
    np.testing.assert_array_equal(buf.get_samples(), np.arange(20, 60, dtype=np.complex64))
    buf.append(np.arange(50, dtype=np.complex64))
    samples = buf.get_samples()
    assert len(samples) == 40
    np.testing.assert_array_equal(samples, np.arange(10, 50, dtype=np.complex64))


