import numpy as np
from .base import DemoduladorBase


class DemoduladorBTLE(DemoduladorBase):
    """
    Demodulador BLE (Bluetooth Low Energy) con:
    - Sincronización por correlación con preámbulo GFSK
    - Estimación y corrección de CFO (Carrier Frequency Offset)
    - Detección de burst por potencia como fallback

    Implementa las funciones del preámbulo BLE según la especificación:
    1. Symbol Timing Recovery (sincronización de reloj mediante correlación)
    2. CFO estimation/correction (la media del preámbulo alternado = CFO)
    3. AGC training (la potencia del preámbulo estabiliza la ganancia)

    El preámbulo BLE es el campo inicial obligatorio de todos los paquetes
    de la Link Layer. Es una secuencia alternada de bits que NO se somete
    a whitening y está fuera de la cobertura del CRC.

    Para LE 1M PHY: 1 octeto (8 bits), pattern según Access Address:
      - Si primer bit físico AA = 0 → preámbulo 01010101 (termina en 1)
      - Si primer bit físico AA = 1 → preámbulo 10101010 (termina en 0)
    Esto garantiza 8 transiciones consecutivas para el enganche de reloj.
    """

    def __init__(self):
        super().__init__()
        self._id = 'btle'
        self._nombre_mostrar = 'BTLE (Bluetooth Low Energy)'
        self.sample_rate = 20e6
        self.fft_size = 2048

        self.buffer_len_s = 0.05
        self.buffer = np.array([], dtype=np.complex64)
        self.last_burst_metrics = None

        # ── Parámetros de la capa física LE 1M ──
        self.bit_rate = 1e6            # 1 Mbps (LE 1M PHY)
        self.modulation_index = 0.5    # h = 0.5
        # Desviación: Δf = h × bit_rate / 2 = ±250 kHz
        self.freq_deviation = self.modulation_index * self.bit_rate / 2
        self.bt_product = 0.5          # BT del filtro Gaussiano GFSK

        # Preámbulo: 8 bits para LE 1M PHY (1 octeto).
        # Para LE 2M sería 16 bits; para LE Coded, 80 símbolos de '00111100'.
        self.preamble_len_bits = 8

        # Variantes del preámbulo según el Access Address.
        # Se generan dinámicamente: el último bit del preámbulo debe
        # ser distinto al primer bit físico (LSB) del Access Address.
        self._preamble_bits_variants = [
            np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float),  # AA LSB=0
            np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float),  # AA LSB=1
        ]

        # Pre-calculados en configurar()
        self._preamble_refs = []
        self._samples_per_bit = None
        self._preamble_len_samples = None

        # Umbral mínimo de correlación normalizada para considerar
        # que se encontró un preámbulo válido. Valores típicos:
        # >0.7 = señal limpia, 0.3-0.7 = señal ruidosa, <0.3 = no hay
        self._sync_threshold = 0.3

    @property
    def id(self) -> str:
        return self._id

    @property
    def nombre_mostrar(self) -> str:
        return self._nombre_mostrar

    # ──────────────────────────────────────────────────────────────────
    # Generación de referencia GFSK para correlación
    # ──────────────────────────────────────────────────────────────────

    def _generate_gfsk_freq_reference(self, bits):
        """
        Genera el patrón ideal de desviación de frecuencia GFSK para
        una secuencia de bits dada.

        BLE usa GFSK con BT=0.5 y h=0.5. La referencia se genera:
        1. Codificando bits en NRZ (0→-1, 1→+1)
        2. Creando un tren de pulsos rectangulares (1 bit = sps muestras)
        3. Aplicando un filtro Gaussiano (BT=0.5) para suavizar transiciones
        4. Escalando a ±250 kHz de desviación

        El resultado es la curva de frecuencia instantánea que un receptor
        ideal vería al recibir esta secuencia de bits.

        Args:
            bits: ndarray de bits (0/1), típicamente 8 bits de preámbulo

        Returns:
            ndarray: Desviación de frecuencia ideal en Hz
        """
        sps = self._samples_per_bit

        # Codificación NRZ: bit 0 → -1, bit 1 → +1
        nrz = 2.0 * bits - 1.0

        # Tren de pulsos rectangulares (cada bit mantenido sps muestras)
        rect_pulse = np.repeat(nrz, sps)

        # ── Filtro Gaussiano para GFSK con BT=0.5 ──
        # El ancho de banda a -3dB del filtro es B = BT/T (T = periodo de bit)
        # La desviación estándar equivalente en tiempo:
        #   σ_t = sqrt(ln2) / (2π × B) = sqrt(ln2) / (2π × BT × bit_rate)
        # En muestras: σ_samples = σ_t × sample_rate
        sigma_bits = np.sqrt(np.log(2)) / (2 * np.pi * self.bt_product)
        sigma_samples = sigma_bits * sps

        # Kernel Gaussiano (±3σ para capturar >99.7% de la energía)
        half_len = int(np.ceil(3 * sigma_samples))
        k = np.arange(-half_len, half_len + 1)
        gaussian = np.exp(-k ** 2 / (2 * sigma_samples ** 2))
        gaussian /= np.sum(gaussian)  # Normalizar

        # Aplicar filtro → transiciones suaves estilo GFSK
        freq_ref = np.convolve(rect_pulse, gaussian, mode='same')

        # Escalar a desviación de frecuencia (±250 kHz)
        freq_ref *= self.freq_deviation

        return freq_ref

    # ──────────────────────────────────────────────────────────────────
    # Detección de preámbulo por correlación cruzada normalizada (NCC)
    # ──────────────────────────────────────────────────────────────────

    def _find_preamble_by_correlation(self, freq_dev_hz, approx_starts):
        """
        Busca el inicio del preámbulo BTLE mediante Correlación Cruzada
        Normalizada (NCC) entre la señal FM-demodulada y las referencias.

        Para evitar falsos positivos (por ejemplo, el patrón 10101010 
        presente aleatoriamente dentro de la carga útil PRBS9), la búsqueda 
        se restringe a una ventana de tiempo cercana al flanco de subida 
        de potencia de la ráfaga (approx_starts).

        Args:
            freq_dev_hz: Desviación de frecuencia instantánea en Hz
            approx_starts: Lista de índices (muestras) donde se detectó el inicio de una ráfaga

        Returns:
            tuple: (start_index, quality, variant_index)
        """
        if not self._preamble_refs or not approx_starts:
            return None, 0.0, 0

        best_start = None
        best_quality = 0.0
        best_variant = 0
        
        # Margen de búsqueda: +/- 15 us alrededor del flanco de subida
        margin = int(self.sample_rate * 15e-6)

        for s in approx_starts:
            search_start = max(0, s - margin)
            search_end = min(len(freq_dev_hz), s + margin)
            window_freq_dev = freq_dev_hz[search_start:search_end]
            
            for variant_idx, ref in enumerate(self._preamble_refs):
                ref_len = len(ref)
                if len(window_freq_dev) < ref_len:
                    continue

                ref_centered = ref - np.mean(ref)
                ref_energy = np.linalg.norm(ref_centered)
                if ref_energy < 1e-12:
                    continue
                ref_norm = ref_centered / ref_energy

                raw_corr = np.correlate(window_freq_dev, ref_norm, mode='valid')
                num_pos = len(raw_corr)

                n = ref_len
                cs = np.empty(len(window_freq_dev) + 1)
                cs[0] = 0.0
                np.cumsum(window_freq_dev, out=cs[1:])

                cs2 = np.empty(len(window_freq_dev) + 1)
                cs2[0] = 0.0
                np.cumsum(window_freq_dev ** 2, out=cs2[1:])

                local_sum = cs[n:n + num_pos] - cs[:num_pos]
                local_sum2 = cs2[n:n + num_pos] - cs2[:num_pos]
                local_mean = local_sum / n
                local_var = local_sum2 / n - local_mean ** 2
                local_energy = np.sqrt(np.maximum(local_var * n, 0)) + 1e-12

                ncc = raw_corr / local_energy
                abs_ncc = np.abs(ncc)
                
                # Buscar máximos locales que superen el umbral
                is_peak = (abs_ncc[1:-1] > abs_ncc[:-2]) & (abs_ncc[1:-1] > abs_ncc[2:])
                is_peak = np.concatenate(([False], is_peak, [False]))
                valid_peaks = np.where(is_peak & (abs_ncc >= self._sync_threshold))[0]
                
                if len(valid_peaks) > 0:
                    # Tomar el PRIMER pico válido en el tiempo, pero cuidado:
                    # debido al ruido, puede haber pequeños falsos "picos locales" (ripples)
                    # en la ladera de subida de la montaña de correlación principal.
                    # Para evitar elegir un ripple de baja calidad (ej. 0.33) en lugar
                    # de la cima real (ej. 0.95), agrupamos todos los picos que ocurren
                    # dentro de 1.5 us (el periodo de repetición del preámbulo es 2 us)
                    # y nos quedamos con el máximo absoluto de ese primer grupo.
                    first_peak = valid_peaks[0]
                    cluster_window = int(self.sample_rate * 1.5e-6)
                    cluster = valid_peaks[valid_peaks - first_peak <= cluster_window]
                    
                    peak_idx_local = cluster[np.argmax(abs_ncc[cluster])]
                    peak_quality = float(abs_ncc[peak_idx_local])
                else:
                    # Si no hay picos válidos, guardamos el máximo absoluto por si acaso
                    peak_idx_local = np.argmax(abs_ncc)
                    peak_quality = float(abs_ncc[peak_idx_local])

                if peak_quality > best_quality:
                    best_quality = peak_quality
                    best_start = search_start + int(peak_idx_local)
                    best_variant = variant_idx

            # Si encontramos un preámbulo válido en esta ráfaga, no buscamos en las siguientes
            if best_quality >= self._sync_threshold:
                return best_start, best_quality, best_variant

        return None, best_quality, best_variant

    # ──────────────────────────────────────────────────────────────────
    # Estimación de CFO desde el preámbulo
    # ──────────────────────────────────────────────────────────────────

    def _estimate_cfo(self, freq_dev_hz, preamble_start):
        """
        Estima el Carrier Frequency Offset (CFO) a partir del preámbulo.

        El preámbulo BLE es una secuencia perfectamente alternada (01010101
        o 10101010). En GFSK con h=0.5, las desviaciones positivas (+Δf)
        y negativas (-Δf) se compensan exactamente entre sí, haciendo que
        la media de la frecuencia instantánea durante el preámbulo sea
        exactamente 0 Hz en ausencia de CFO.

        Por lo tanto:
            CFO_estimado = mean(freq_dev_hz[preámbulo])

        Este es el método estándar usado en receptores BLE reales para
        compensar el desplazamiento de portadora antes de decodificar
        los datos del paquete.

        Args:
            freq_dev_hz: Desviación de frecuencia instantánea en Hz
            preamble_start: Índice de inicio del preámbulo

        Returns:
            float: CFO estimado en Hz
        """
        preamble_end = min(
            preamble_start + self._preamble_len_samples,
            len(freq_dev_hz)
        )

        preamble_region = freq_dev_hz[preamble_start:preamble_end]

        if len(preamble_region) == 0:
            return 0.0

        return float(np.mean(preamble_region))

    # ──────────────────────────────────────────────────────────────────
    # Configuración
    # ──────────────────────────────────────────────────────────────────

    def configurar(self, sample_rate: float, fft_size: int,
                   bw_mhz: float = 1):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bw_mhz = bw_mhz
        self.buffer = np.array([], dtype=np.complex64)

        # Calcular parámetros derivados del sample rate
        self._samples_per_bit = int(self.sample_rate / self.bit_rate)
        self._preamble_len_samples = (self.preamble_len_bits
                                      * self._samples_per_bit)

        # Pre-generar las referencias de desviación de frecuencia GFSK
        # para ambas variantes de preámbulo. Se computan una sola vez
        # aquí y se reusan en cada llamada a procesar().
        self._preamble_refs = [
            self._generate_gfsk_freq_reference(bits)
            for bits in self._preamble_bits_variants
        ]

    # ──────────────────────────────────────────────────────────────────
    # Procesamiento principal
    # ──────────────────────────────────────────────────────────────────

    def procesar(self, muestras_iq: np.ndarray) -> dict:
        self.buffer = np.concatenate((self.buffer, muestras_iq))

        target_len = int(self.sample_rate * self.buffer_len_s)
        resultados = {}

        if len(self.buffer) >= target_len:
            iq_samples = self.buffer[:target_len]
            overlap = int(self.sample_rate * 2e-3)
            self.buffer = self.buffer[target_len - overlap:]

            # ═════════════════════════════════════════════════════════
            # PASO 1: FM Demodulación del buffer completo
            # Derivada de la fase → frecuencia instantánea
            # ═════════════════════════════════════════════════════════
            phase = np.unwrap(np.angle(iq_samples))
            freq_dev_hz = np.diff(phase) / (2 * np.pi) * self.sample_rate
            freq_dev_hz = np.append(freq_dev_hz, freq_dev_hz[-1])
            
            # Limitar matemáticamente los picos transitorios de discontinuidad de fase.
            # BLE usa desviación de +-250 kHz. Limitando a +-800 kHz damos muchísimo
            # margen para el CFO (desalineación de portadora), pero matamos los picos
            # de encendido/ruido que llegan a 5-10 MHz y rompen el auto-scale del gráfico.
            freq_dev_hz = np.clip(freq_dev_hz, -800000, 800000)

            # ═════════════════════════════════════════════════════════
            # PASO 2: Detección aproximada de ráfagas (potencia)
            # ═════════════════════════════════════════════════════════
            power = np.abs(iq_samples) ** 2
            window_size = max(1, int(self.sample_rate * 50e-6))
            window = np.ones(window_size) / window_size
            smoothed_power = np.convolve(power, window, mode='same')
            
            p_min = np.min(smoothed_power)
            p_max = np.max(smoothed_power)
            dynamic_range_db = 10 * np.log10((p_max + 1e-20) / (p_min + 1e-20))
            
            approx_starts = []
            if dynamic_range_db >= 6:
                threshold = p_min + (p_max - p_min) * 0.2
                is_active = smoothed_power > threshold
                edges = np.diff(is_active.astype(int))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                
                if len(is_active) > 0 and is_active[0]:
                    starts = np.insert(starts, 0, 0)
                if len(is_active) > 0 and is_active[-1]:
                    ends = np.append(ends, len(iq_samples) - 1)
                    
                min_burst_len = int(self.sample_rate * 20e-6)
                for s, e in zip(starts, ends):
                    if ((e - s) > min_burst_len and s > 0 and e < len(iq_samples) - 1):
                        approx_starts.append(s)

            # ═════════════════════════════════════════════════════════
            # PASO 3: Sincronización por correlación con preámbulo
            # El preámbulo cumple la función de Symbol Timing Recovery:
            # la NCC con la referencia GFSK localiza exactamente dónde
            # empieza cada símbolo (duración 1 µs en LE 1M).
            # ═════════════════════════════════════════════════════════
            preamble_start, sync_quality, preamble_variant = \
                self._find_preamble_by_correlation(freq_dev_hz, approx_starts)

            extract_start = None
            extract_end = None
            cfo_hz = 0.0
            preamble_found = (preamble_start is not None)

            if preamble_found:
                # ═════════════════════════════════════════════════════
                # PASO 3: Estimación de CFO desde el preámbulo
                # Al ser alternado, las desviaciones +/- se cancelan,
                # y cualquier offset residual es el CFO.
                # ═════════════════════════════════════════════════════
                cfo_hz = self._estimate_cfo(freq_dev_hz, preamble_start)

                # ═════════════════════════════════════════════════════
                # PASO 4: Corrección de CFO
                # Restar el offset de toda la señal FM-demodulada
                # para que los niveles ±250 kHz queden centrados en 0.
                # ═════════════════════════════════════════════════════
                freq_dev_hz -= cfo_hz

                # ═════════════════════════════════════════════════════
                # PASO 5: Ventana de visualización anclada al preámbulo
                # Al anclar el display al punto de sincronización, la
                # duración y posición de los gráficos son estables
                # entre frames (elimina el "salto" de ventana).
                # ═════════════════════════════════════════════════════
                margin_before = int(self.sample_rate * 10e-6)
                extract_start = max(0, preamble_start - margin_before)

                # 500 µs de ventana (paquete BLE máx ~376 µs)
                max_display = int(self.sample_rate * 500e-6)
                extract_end = min(len(iq_samples),
                                  extract_start + max_display)
            else:
                # ── Fallback: detección por envolvente de potencia ──
                # Para señales sin preámbulo BLE válido (CW, tono, etc.)
                extract_start, extract_end, cfo_hz = \
                    self._fallback_power_detection(
                        iq_samples, freq_dev_hz)
                if cfo_hz != 0.0:
                    freq_dev_hz -= cfo_hz

            # ═════════════════════════════════════════════════════════
            # PASO 6: Cálculo de métricas de la ventana extraída
            # ═════════════════════════════════════════════════════════
            if extract_start is not None:
                burst_samples = iq_samples[extract_start:extract_end]
                n_samples = len(burst_samples)
                burst_time_us = (np.arange(n_samples)
                                 / self.sample_rate * 1e6)

                # Potencia vs Tiempo
                power_mw = np.abs(burst_samples) ** 2
                power_dbm = 10 * np.log10(power_mw + 1e-12)

                # Desviación de frecuencia (ya corregida por CFO)
                freq_dev_khz = (
                    freq_dev_hz[extract_start:extract_end] / 1000.0)

                # Garantizar longitudes consistentes
                min_len = min(len(burst_time_us), len(freq_dev_khz),
                              len(power_dbm))
                burst_time_us = burst_time_us[:min_len]
                power_dbm = power_dbm[:min_len]
                freq_dev_khz = freq_dev_khz[:min_len]

                # Aplicar Squelch (Silenciador) al gráfico de frecuencia
                # Equipos de laboratorio como el CMW500 silencian el trazo de FM 
                # fuera de la ráfaga de energía para limpiar el gráfico.
                # Al ser BLE de envolvente constante, forzamos a 0 kHz 
                # todo lo que esté 10 dB por debajo del pico máximo de potencia.
                peak_pwr_dbm = np.max(power_dbm)
                squelch_mask = power_dbm < (peak_pwr_dbm - 10.0)
                freq_dev_khz[squelch_mask] = 0.0

                channel_offsets_ch = np.array([])
                channel_power_dbm = []
                avg_pwr = 0.0
                peak_pwr = 0.0
                papr = 0.0
                leakage_pwr = -100.0
                
                if not getattr(self, 'skip_metrics', False):
                    # Espectro ACP (Adjacent Channel Power)
                    N_b = len(burst_samples)
                    fft_vals = (np.fft.fftshift(np.fft.fft(burst_samples)) / N_b)
                    power_spectrum_b = np.abs(fft_vals) ** 2
                    freqs_b = np.fft.fftshift(np.fft.fftfreq(N_b, 1 / self.sample_rate))
    
                    channel_bw = getattr(self, 'bw_mhz', 1) * 1e6
                    offsets_mhz = np.arange(-10, 11)
                    channel_offsets_ch = offsets_mhz / 2.0
                    for offset_mhz in offsets_mhz:
                        center_f = offset_mhz * 1e6
                        idx = np.where(
                            (freqs_b >= center_f - channel_bw / 2) &
                            (freqs_b <= center_f + channel_bw / 2))[0]
                        if len(idx) > 0:
                            pwr = np.sum(power_spectrum_b[idx])
                            pwr_dbm = 10 * np.log10(pwr + 1e-12)
                        else:
                            pwr_dbm = -100
                        channel_power_dbm.append(float(pwr_dbm))
    
                    # Calcular métricas de potencia (sobre la parte activa de la ráfaga)
                    peak_pwr = float(np.max(power_dbm))
                    active_mask = power_dbm > (peak_pwr - 10.0)
                    if np.any(active_mask):
                        active_power_mw = power_mw[active_mask]
                        avg_pwr = float(10 * np.log10(np.mean(active_power_mw) + 1e-12))
                    else:
                        avg_pwr = peak_pwr
                    papr = peak_pwr - avg_pwr
                    
                    # Leakage Power: Calculado a partir de los márgenes de silencio de la ráfaga extraída.
                    # Esto evita que otras ráfagas en el buffer grande rompan la medición.
                    burst_power_mw = np.abs(burst_samples)**2
                    burst_power_dbm = 10 * np.log10(burst_power_mw + 1e-12)
                    idle_mask_burst = burst_power_dbm < (peak_pwr - 20.0)
                    
                    if np.any(idle_mask_burst):
                        leakage_pwr = float(10 * np.log10(np.mean(burst_power_mw[idle_mask_burst]) + 1e-12))
                    else:
                        leakage_pwr = -100.0

                self.last_burst_metrics = {
                    'burst_time_us': burst_time_us,
                    'power_dbm': power_dbm,
                    'mag_linear': np.abs(burst_samples)[:min_len],
                    'freq_dev_khz': freq_dev_khz,
                    'acp_channels': channel_offsets_ch,
                    'acp_power_dbm': np.array(channel_power_dbm),
                    # ── Métricas de sincronización y CFO ──
                    'cfo_khz': cfo_hz / 1000.0,
                    'sync_quality': sync_quality,
                    'preamble_found': preamble_found,
                    # ── Métricas de Potencia ──
                    'avg_power_dbm': avg_pwr,
                    'peak_power_dbm': peak_pwr,
                    'papr_db': papr,
                    'leakage_power_dbm': leakage_pwr,
                    'skip_metrics': getattr(self, 'skip_metrics', False)
                }


        fft_data = np.fft.fftshift(
            np.fft.fft(muestras_iq, n=self.fft_size))
        psd = 10 * np.log10(np.abs(fft_data) ** 2 + 1e-12)

        resultados['psd_rf'] = psd
        resultados['rf_chunk'] = np.array([])
        if self.last_burst_metrics is not None:
            resultados['metricas'] = {
                'btle_metrics': self.last_burst_metrics
            }

        return resultados

    # ──────────────────────────────────────────────────────────────────
    # Fallback: detección por envolvente de potencia
    # ──────────────────────────────────────────────────────────────────

    def _fallback_power_detection(self, iq_samples, freq_dev_hz):
        """
        Método de respaldo para detectar la región de interés cuando
        no se encuentra un preámbulo BLE válido por correlación.

        Usa la envolvente de potencia suavizada para distinguir entre:
        - Señal continua (rango dinámico < 6 dB): ventana centrada fija
        - Señal con bursts: detección por umbral de potencia

        En ambos casos estima CFO como la media de freq_dev en la
        región detectada (asumiendo datos balanceados).

        Args:
            iq_samples: Muestras IQ del buffer
            freq_dev_hz: Desviación de frecuencia ya calculada

        Returns:
            tuple: (extract_start, extract_end, cfo_hz)
                   extract_start puede ser None si no se detecta nada
        """
        power = np.abs(iq_samples) ** 2
        window_size = max(1, int(self.sample_rate * 50e-6))
        window = np.ones(window_size) / window_size
        smoothed_power = np.convolve(power, window, mode='same')

        p_min = np.min(smoothed_power)
        p_max = np.max(smoothed_power)
        dynamic_range_db = 10 * np.log10(
            (p_max + 1e-20) / (p_min + 1e-20))

        if dynamic_range_db < 6:
            # Señal continua: ventana centrada de tamaño fijo
            max_display = int(self.sample_rate * 500e-6)
            center = len(iq_samples) // 2
            half = min(max_display // 2, center)
            extract_start = center - half
            extract_end = center + half

            # CFO de señal continua (datos balanceados → media ≈ 0)
            cfo_hz = float(np.mean(
                freq_dev_hz[extract_start:extract_end]))
            return extract_start, extract_end, cfo_hz
        else:
            # Señal con bursts: umbral de potencia
            threshold = p_min + (p_max - p_min) * 0.2
            is_active = smoothed_power > threshold
            edges = np.diff(is_active.astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]

            if len(is_active) > 0 and is_active[0]:
                starts = np.insert(starts, 0, 0)
            if len(is_active) > 0 and is_active[-1]:
                ends = np.append(ends, len(iq_samples) - 1)

            min_burst_len = int(self.sample_rate * 20e-6)
            for s, e in zip(starts, ends):
                if ((e - s) > min_burst_len and
                        s > 0 and e < len(iq_samples) - 1):
                    margin = int(self.sample_rate * 50e-6)
                    extract_start = max(0, s - margin)
                    extract_end = min(len(iq_samples), e + margin)

                    # CFO del burst (datos balanceados → media ≈ 0)
                    cfo_hz = float(np.mean(
                        freq_dev_hz[extract_start:extract_end]))
                    return extract_start, extract_end, cfo_hz

        return None, None, 0.0
