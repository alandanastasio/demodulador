"""Sincronización de ráfagas LE 1M y LE 2M a la tasa de la HackRF."""

import numpy as np
import pytest

from dsp.demoduladores.btle import DemoduladorBTLE


@pytest.mark.parametrize("phy_mbps", [1, 2])
def test_preamble_is_found_after_smoothed_power_edge(phy_mbps):
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048, bw_mhz=phy_mbps)
    demod.skip_metrics = True

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
    assert metrics["sync_quality"] > 0.8
    assert abs(metrics["cfo_khz"] - 50) < 25
    assert "metricas" not in demod.procesar(np.zeros(1024, dtype=np.complex64))


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


def test_silence_clears_packet_metrics():
    demod = DemoduladorBTLE()
    demod.configurar(20_000_000, 2048)
    count = int(demod.sample_rate * demod.buffer_len_s)
    result = demod.procesar(np.zeros(count, dtype=np.complex64))

    assert result["metricas"]["btle_metrics"] is None
