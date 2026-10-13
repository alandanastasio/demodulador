"""Sincronización de ráfagas LE 1M y LE 2M a la tasa de la HackRF."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dsp.demoduladores.btle import DemoduladorBTLE
from plot_renderer import render_plot


@pytest.mark.parametrize("phy_mbps", [1, 2])
def test_preamble_is_found_after_smoothed_power_edge(phy_mbps):
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048, bw_mhz=phy_mbps)

    rng = np.random.default_rng(42)
    count = int(demod.sample_rate * demod.buffer_len_s)
    iq = (rng.normal(0, 0.005, count)
          + 1j * rng.normal(0, 0.005, count)).astype(np.complex64)

    bits = np.concatenate((
        demod._preamble_bits_variants[0],
        rng.integers(0, 2, 80 * phy_mbps),
    ))
    frequency = demod._generate_gfsk_freq_reference(bits) + 50_000
    burst = np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
    burst_start = 100_000
    iq[burst_start:burst_start + len(burst)] += 0.7 * burst
    iq += np.complex64(0.03 + 0.02j)

    metrics = demod.procesar(iq)["metricas"]["btle_metrics"]

    assert demod.preamble_len_bits == 8 * phy_mbps
    assert metrics["preamble_found"] is True
    assert metrics["sync_quality"] > 0.95
    assert abs(metrics["cfo_khz"] - 50) < 25
    # El GFSK generado nunca excede la desviación nominal: el filtro de canal
    # no debe crear picos grandes al atravesar la frecuencia central.
    assert metrics["df1_max_khz"] < 350 * phy_mbps
    assert metrics["df2_min_khz"] > -350 * phy_mbps
    window_power_db = 10 * np.log10(np.mean(metrics["mag_linear"] ** 2))
    # Ninguna banda de la FFT puede contener más potencia que toda la ventana.
    assert np.max(metrics["acp_power_dbm"]) <= window_power_db + 0.1
    assert "metricas" not in demod.procesar(np.zeros(1024, dtype=np.complex64))


def test_weak_but_consistent_gfsk_preamble_is_kept():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    demod.skip_metrics = True
    rng = np.random.default_rng(42)
    count = int(demod.sample_rate * demod.buffer_len_s)
    iq = (rng.normal(0, 0.005, count)
          + 1j * rng.normal(0, 0.005, count)).astype(np.complex64)
    bits = np.concatenate((demod._preamble_bits_variants[0],
                           rng.integers(0, 2, 300)))
    frequency = demod._generate_gfsk_freq_reference(bits)
    burst = np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
    iq[100_000:100_000 + len(burst)] += 0.015 * burst
    iq += np.complex64(0.016 + 0.005j)

    metrics = demod.procesar(iq)["metricas"]["btle_metrics"]

    assert metrics is not None
    assert demod._WEAK_SYNC_QUALITY < metrics["sync_quality"] < demod._STRONG_SYNC_QUALITY


def test_non_ble_burst_does_not_hide_later_packet():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    demod.skip_metrics = True

    rng = np.random.default_rng(42)
    count = int(demod.sample_rate * demod.buffer_len_s)
    iq = (rng.normal(0, 0.005, count)
          + 1j * rng.normal(0, 0.005, count)).astype(np.complex64)
    tone_phase = 2 * np.pi * 300_000 * np.arange(4000) / demod.sample_rate
    iq[20_000:24_000] += 0.7 * np.exp(1j * tone_phase)

    bits = np.concatenate((demod._preamble_bits_variants[0],
                           rng.integers(0, 2, 80)))
    frequency = demod._generate_gfsk_freq_reference(bits) + 50_000
    burst = np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
    iq[100_000:100_000 + len(burst)] += 0.7 * burst

    metrics = demod.procesar(iq)["metricas"]["btle_metrics"]

    assert metrics["preamble_found"] is True
    assert metrics["sync_quality"] > 0.8
    assert abs(metrics["cfo_khz"] - 50) < 25


@pytest.mark.parametrize("phy_mbps", [1, 2])
def test_long_burst_is_not_cut_at_500_us(phy_mbps):
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048, bw_mhz=phy_mbps)
    demod.skip_metrics = True

    rng = np.random.default_rng(23)
    count = int(demod.sample_rate * demod.buffer_len_s)
    iq = (rng.normal(0, 0.005, count)
          + 1j * rng.normal(0, 0.005, count)).astype(np.complex64)
    bits = np.concatenate((demod._preamble_bits_variants[0],
                           rng.integers(0, 2, 900 * phy_mbps)))
    frequency = demod._generate_gfsk_freq_reference(bits)
    burst = np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
    start = 100_000
    iq[start:start + len(burst)] += 0.7 * burst

    metrics = demod.procesar(iq)["metricas"]["btle_metrics"]

    assert metrics["preamble_found"] is True
    assert metrics["burst_time_us"][-1] > 900
    assert metrics["burst_time_us"][-1] < 1_000


def test_long_burst_crossing_processing_boundary_is_found():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    demod.skip_metrics = True

    rng = np.random.default_rng(24)
    count = int(demod.sample_rate * 0.098)
    iq = (rng.normal(0, 0.005, count)
          + 1j * rng.normal(0, 0.005, count)).astype(np.complex64)
    bits = np.concatenate((demod._preamble_bits_variants[0],
                           rng.integers(0, 2, 2_500)))
    frequency = demod._generate_gfsk_freq_reference(bits)
    burst = np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
    start = int(demod.sample_rate * 0.0475)
    iq[start:start + len(burst)] += 0.7 * burst

    block = int(demod.sample_rate * demod.buffer_len_s)
    demod.procesar(iq[:block])
    metrics = demod.procesar(iq[block:])["metricas"]["btle_metrics"]

    assert metrics["preamble_found"] is True
    assert metrics["burst_time_us"][-1] > 2_500


@pytest.mark.parametrize("phy_mbps", [1, 2])
@pytest.mark.parametrize("start_us", [0, 5, 10])
def test_first_burst_at_capture_boundary_is_found(phy_mbps, start_us):
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048, bw_mhz=phy_mbps)
    demod.skip_metrics = True

    rng = np.random.default_rng(33)
    count = int(demod.sample_rate * demod.buffer_len_s)
    iq = (rng.normal(0, 0.005, count)
          + 1j * rng.normal(0, 0.005, count)).astype(np.complex64)
    bits = np.concatenate((demod._preamble_bits_variants[0],
                           rng.integers(0, 2, 200 * phy_mbps)))
    frequency = demod._generate_gfsk_freq_reference(bits)
    burst = np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
    start = int(demod.sample_rate * start_us * 1e-6)
    iq[start:start + len(burst)] += 0.7 * burst

    metrics = demod.procesar(iq)["metricas"]["btle_metrics"]

    assert metrics["preamble_found"] is True
    assert metrics["sync_quality"] > 0.9


def test_silence_does_not_emit_packet_metrics():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    count = int(demod.sample_rate * demod.buffer_len_s)
    result = demod.procesar(np.zeros(count, dtype=np.complex64))

    assert result["metricas"]["btle_metrics"] is None


@pytest.mark.parametrize("phy_mbps", [1, 2])
def test_continuous_gfsk_does_not_replace_a_packet(phy_mbps):
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048, bw_mhz=phy_mbps)
    count = int(demod.sample_rate * demod.buffer_len_s)
    rng = np.random.default_rng(34)
    bits = rng.integers(0, 2, 200 * phy_mbps)
    frequency = np.resize(demod._generate_gfsk_freq_reference(bits), count)
    iq = (0.3 * np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
          + 0.005 * (rng.normal(size=count) + 1j * rng.normal(size=count)))

    result = demod.procesar(iq.astype(np.complex64))

    assert result["metricas"]["btle_metrics"] is None


@pytest.mark.parametrize("next_signal", ["silence", "continuous", "tone_burst"])
def test_last_packet_is_retained_without_a_new_frame(next_signal):
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    demod.skip_metrics = True
    rng = np.random.default_rng(36)
    count = int(demod.sample_rate * demod.buffer_len_s)
    first_iq = (rng.normal(0, 0.005, count)
                + 1j * rng.normal(0, 0.005, count)).astype(np.complex64)
    bits = np.concatenate((demod._preamble_bits_variants[0],
                           rng.integers(0, 2, 200)))
    frequency = demod._generate_gfsk_freq_reference(bits)
    burst = np.exp(2j * np.pi * np.cumsum(frequency) / demod.sample_rate)
    first_iq[100_000:100_000 + len(burst)] += 0.7 * burst
    first = demod.procesar(first_iq)
    previous = demod.last_burst_metrics

    next_iq = np.zeros(count, dtype=np.complex64)
    if next_signal == "continuous":
        next_iq = (0.3 * np.exp(2j * np.pi * 300_000
                               * np.arange(count) / demod.sample_rate)).astype(np.complex64)
    elif next_signal == "tone_burst":
        phase = 2 * np.pi * 300_000 * np.arange(4_000) / demod.sample_rate
        next_iq[20_000:24_000] = 0.7 * np.exp(1j * phase)
    following = demod.procesar(next_iq)

    assert first["metricas"]["btle_metrics"] is previous
    assert previous["preamble_found"] is True
    assert following["metricas"]["btle_metrics"] is None
    assert demod.last_burst_metrics is previous
    assert not np.array_equal(first["psd_rf"], following["psd_rf"])


@pytest.mark.parametrize("interference", ["noise", "dc", "adjacent_tone"])
def test_no_continuous_btle_signal_for_noise_dc_or_adjacent_channel(interference):
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    count = int(demod.sample_rate * demod.buffer_len_s)
    rng = np.random.default_rng(35)
    iq = (rng.normal(0, 0.005, count)
          + 1j * rng.normal(0, 0.005, count))
    if interference == "dc":
        iq += 0.3 + 0.2j
    elif interference == "adjacent_tone":
        iq += 0.3 * np.exp(2j * np.pi * 4e6 * np.arange(count) / demod.sample_rate)

    result = demod.procesar(iq.astype(np.complex64))

    assert result["metricas"]["btle_metrics"] is None


def test_tone_transient_is_not_a_btle_preamble():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    demod.skip_metrics = True
    count = int(demod.sample_rate * demod.buffer_len_s)
    rng = np.random.default_rng(42)
    iq = (rng.normal(0, 0.005, count)
          + 1j * rng.normal(0, 0.005, count)).astype(np.complex64)
    phase = 2 * np.pi * 300_000 * np.arange(4000) / demod.sample_rate
    iq[20_000:24_000] += 0.7 * np.exp(1j * phase)

    result = demod.procesar(iq)

    assert result["metricas"]["btle_metrics"] is None


def test_renderer_keeps_previous_packet_but_updates_spectrum():
    psd = np.array([-40.0, -30.0])
    view = SimpleNamespace(
        is_paused=False,
        f_axis=np.array([2401.0, 2402.0]),
        freq_plot_curve=Mock(),
        btle_freq_curve=Mock(),
        btle_power_curve=Mock(),
        btle_mag_curve=Mock(),
        btle_acp_bars=Mock(),
        btle_metrics_label=Mock(),
        trace_manager=SimpleNamespace(process=Mock(return_value=psd)),
        marker_manager=SimpleNamespace(update_render=Mock()),
    )

    render_plot(view, {"demod_mode": "btle"}, psd, None,
                fm_metrics={"btle_metrics": None})

    view.freq_plot_curve.setData.assert_called_once()
    np.testing.assert_array_equal(view.freq_plot_curve.setData.call_args.args[1], psd)
    view.btle_freq_curve.setData.assert_not_called()
    view.btle_power_curve.setData.assert_not_called()
    view.btle_mag_curve.setData.assert_not_called()
    view.btle_acp_bars.setOpts.assert_not_called()
    view.btle_metrics_label.setText.assert_not_called()
