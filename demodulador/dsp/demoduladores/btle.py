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

    # ── Constantes ──
    _BURST_ACTIVATION_RATIO = 0.2       # Umbral de activación (20% del rango dinámico)
    _MIN_DYNAMIC_RANGE_DB = 6           # Rango dinámico mínimo para considerar burst (dB)
    _SQUELCH_THRESHOLD_DB = 10.0        # Squelch: dB por debajo del pico para silenciar FM
    _LEAKAGE_THRESHOLD_DB = 20.0        # Leakage: dB por debajo del pico para medir fuga
    _FREQ_CLAMP_HZ = 800000             # Clamp de frecuencia instantánea (Hz)
    _DISPLAY_WINDOW_S = 500e-6          # Ventana de visualización (s) — paquete BLE máx ~376 µs

    def __init__(self):
        super().__init__()
        self._id = 'btle'
        self._nombre_mostrar = 'BTLE (Bluetooth Low Energy)'
        self.sample_rate = 20e6
        self.fft_size = 2048

        self.buffer_len_s = 0.05
        self.buffer = np.array([], dtype=np.complex64)
        self.last_burst_metrics = None
        self.skip_metrics = False

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
        self._sync_threshold = 0.35

        # Límite máximo del buffer (3x el target para absorber jitter)
        self._max_buffer_len = None  # Se calcula en configurar()

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
    # Detección de ráfagas por envolvente de potencia
    # ──────────────────────────────────────────────────────────────────

    def _detect_bursts(self, iq_samples):
        """
        Detecta ráfagas en las muestras IQ usando la envolvente de
        potencia suavizada.

        Args:
            iq_samples: Muestras IQ del buffer

        Returns:
            tuple: (starts, ends, smoothed_power, dynamic_range_db)
                   starts/ends son arrays de índices de inicio/fin de ráfagas
        """
        power = np.abs(iq_samples) ** 2
        window_size = max(1, int(self.sample_rate * 50e-6))
        window = np.ones(window_size) / window_size
        smoothed_power = np.convolve(power, window, mode='same')

        p_min = np.min(smoothed_power)
        p_max = np.max(smoothed_power)
        dynamic_range_db = 10 * np.log10((p_max + 1e-20) / (p_min + 1e-20))

        starts = np.array([], dtype=int)
        ends = np.array([], dtype=int)

        if dynamic_range_db >= self._MIN_DYNAMIC_RANGE_DB:
            threshold = p_min + (p_max - p_min) * self._BURST_ACTIVATION_RATIO
            is_active = smoothed_power > threshold
            edges = np.diff(is_active.astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]

            if len(is_active) > 0 and is_active[0]:
                starts = np.insert(starts, 0, 0)
            if len(is_active) > 0 and is_active[-1]:
                ends = np.append(ends, len(iq_samples) - 1)

            # Filtrar ráfagas demasiado cortas o pegadas a los bordes
            min_burst_len = int(self.sample_rate * 20e-6)
            valid = []
            for i, (s, e) in enumerate(zip(starts, ends)):
                if (e - s) > min_burst_len and s > 0 and e < len(iq_samples) - 1:
                    valid.append(i)
            if valid:
                starts = starts[valid]
                ends = ends[valid]
            else:
                starts = np.array([], dtype=int)
                ends = np.array([], dtype=int)

        return starts, ends, smoothed_power, dynamic_range_db

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
        if not self._preamble_refs or len(approx_starts) == 0:
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
                    # No se encontraron picos válidos por encima del umbral.
                    # Ignoramos esta variante en vez de tomar el argmax del ruido,
                    # que generaría falsos positivos con quality ~0.2-0.4.
                    continue

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

        # Límite máximo del buffer (3x el target para absorber jitter)
        self._max_buffer_len = int(self.sample_rate * self.buffer_len_s * 3)

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

        # Protección contra crecimiento indefinido del buffer
        if self._max_buffer_len and len(self.buffer) > self._max_buffer_len:
            self.buffer = self.buffer[-self._max_buffer_len:]

        target_len = int(self.sample_rate * self.buffer_len_s)
        resultados = {}

        if len(self.buffer) >= target_len:
            iq_samples = self.buffer[:target_len]
            overlap = int(self.sample_rate * 2e-3)
            self.buffer = self.buffer[target_len - overlap:]

            # Remover DC Offset (Fuga del Oscilador Local / LO Leakage). 
            # Esto evita oscilaciones de baja frecuencia (wobble/senoidal) en la
            # demodulación de FM que ocurren cuando hay un DC Offset y un pequeño CFO.
            iq_samples = iq_samples - np.mean(iq_samples)

            # ═════════════════════════════════════════════════════════
            # PASO 1: FM Demodulación del buffer completo
            # Derivada de la fase → frecuencia instantánea
            # ═════════════════════════════════════════════════════════
            phase = np.unwrap(np.angle(iq_samples))
            freq_dev_hz = np.diff(phase) / (2 * np.pi) * self.sample_rate
            # Igualar longitud al array original duplicando la última muestra
            freq_dev_hz = np.concatenate((freq_dev_hz, freq_dev_hz[-1:]))
            
            # Limitar matemáticamente los picos transitorios de discontinuidad de fase.
            # BLE usa desviación de +-250 kHz. Limitando a +-800 kHz damos muchísimo
            # margen para el CFO (desalineación de portadora), pero matamos los picos
            # de encendido/ruido que llegan a 5-10 MHz y rompen el auto-scale del gráfico.
            np.clip(freq_dev_hz, -self._FREQ_CLAMP_HZ, self._FREQ_CLAMP_HZ, out=freq_dev_hz)

            # ═════════════════════════════════════════════════════════
            # PASO 2: Detección aproximada de ráfagas (potencia)
            # ═════════════════════════════════════════════════════════
            burst_starts, burst_ends, _, _ = self._detect_bursts(iq_samples)

            # ═════════════════════════════════════════════════════════
            # PASO 3: Sincronización por correlación con preámbulo
            # El preámbulo cumple la función de Symbol Timing Recovery:
            # la NCC con la referencia GFSK localiza exactamente dónde
            # empieza cada símbolo (duración 1 µs en LE 1M).
            # ═════════════════════════════════════════════════════════
            preamble_start, sync_quality, preamble_variant = \
                self._find_preamble_by_correlation(freq_dev_hz, burst_starts)

            extract_start = None
            extract_end = None
            cfo_hz = 0.0
            preamble_found = (preamble_start is not None)

            if preamble_found:
                # ─── Estimación de CFO desde el preámbulo ───
                # Al ser alternado, las desviaciones +/- se cancelan,
                # y cualquier offset residual es el CFO.
                cfo_hz = self._estimate_cfo(freq_dev_hz, preamble_start)

                # ─── Corrección de CFO ───
                # Restar el offset de toda la señal FM-demodulada
                # para que los niveles ±250 kHz queden centrados en 0.
                freq_dev_hz -= cfo_hz

                # ─── Ventana de visualización anclada al preámbulo ───
                # Al anclar el display al punto de sincronización, la
                # duración y posición de los gráficos son estables
                # entre frames (elimina el "salto" de ventana).
                margin_before = int(self.sample_rate * 10e-6)
                extract_start = max(0, preamble_start - margin_before)

                max_display = int(self.sample_rate * self._DISPLAY_WINDOW_S)
                extract_end = min(len(iq_samples),
                                  extract_start + max_display)
            else:
                # ── Fallback: detección por envolvente de potencia ──
                # Para señales sin preámbulo BLE válido (CW, tono, etc.)
                extract_start, extract_end, cfo_hz = \
                    self._fallback_power_detection(
                        iq_samples, freq_dev_hz, burst_starts, burst_ends)
                if cfo_hz != 0.0:
                    freq_dev_hz -= cfo_hz

            # ═════════════════════════════════════════════════════════
            # PASO 4: Cálculo de métricas de la ventana extraída
            # ═════════════════════════════════════════════════════════
            if extract_start is not None:
                burst_samples = iq_samples[extract_start:extract_end]
                
                # RE-DEMODULAR LA RÁFAGA LOCALMENTE PARA ELIMINAR EL WOBBLE (DC OFFSET DINÁMICO)
                # El AGC puede cambiar el DC offset durante la ráfaga. Restar la media no es perfecto
                # si la ráfaga es corta y tiene CFO. Usamos min/max para hallar el centro real (señal de envolvente constante).
                I = np.real(burst_samples)
                Q = np.imag(burst_samples)
                center_I = (np.max(I) + np.min(I)) / 2.0
                center_Q = (np.max(Q) + np.min(Q)) / 2.0
                burst_centered = burst_samples - (center_I + 1j * center_Q)
                
                b_phase = np.unwrap(np.angle(burst_centered))
                b_freq_dev_hz = np.diff(b_phase) / (2 * np.pi) * self.sample_rate
                b_freq_dev_hz = np.concatenate((b_freq_dev_hz, b_freq_dev_hz[-1:]))
                
                # Limitar los picos impulsivos antes del filtro (Spike Killer)
                # Un salto de fase por ruido a 20 Msps genera picos irreales de +-10 MHz.
                # Si entran al filtro sin limitar, su energía "ensancha" el filtro hasta +-500 kHz.
                np.clip(b_freq_dev_hz, -500000.0, 500000.0, out=b_freq_dev_hz)
                
                # Filtrar el ruido de alta frecuencia (Suavizado FM)
                # Aplicamos un filtro pasabajos Butterworth ajustado al ancho de banda real
                # de la señal GFSK (BT=0.5 -> ~500 kHz). Esto es crucial en tráfico vivo (bajo SNR) 
                # porque la derivada de fase (FM) amplifica exponencialmente el ruido térmico.
                from scipy.signal import butter, lfilter
                nyq = 0.5 * self.sample_rate
                cutoff = 0.5e6 / nyq  # 500 kHz cutoff
                b, a = butter(4, cutoff, btype='low')
                b_freq_dev_hz = lfilter(b, a, b_freq_dev_hz)
                
                # Aplicamos la misma corrección de CFO que se calculó en el Paso 3
                if cfo_hz != 0.0:
                    b_freq_dev_hz -= cfo_hz

                n_samples = len(burst_samples)
                burst_time_us = (np.arange(n_samples)
                                 / self.sample_rate * 1e6)

                # Potencia vs Tiempo
                power_mw = np.abs(burst_samples) ** 2
                power_dbm = 10 * np.log10(power_mw + 1e-12)

                # Desviación de frecuencia (re-calculada y corregida)
                freq_dev_khz = b_freq_dev_hz / 1000.0

                # Garantizar longitudes consistentes
                min_len = min(len(burst_time_us), len(freq_dev_khz),
                              len(power_dbm))
                burst_time_us = burst_time_us[:min_len]
                power_dbm = power_dbm[:min_len]
                power_mw = power_mw[:min_len]
                freq_dev_khz = freq_dev_khz[:min_len]

                # Aplicar Squelch (Silenciador) al gráfico de frecuencia
                # Equipos de laboratorio como el CMW500 silencian el trazo de FM 
                # fuera de la ráfaga de energía para limpiar el gráfico.
                # Al ser BLE de envolvente constante, forzamos a 0 kHz 
                # todo lo que esté por debajo del umbral de squelch.
                peak_pwr_dbm = np.max(power_dbm)
                squelch_mask = power_dbm < (peak_pwr_dbm - self._SQUELCH_THRESHOLD_DB)
                freq_dev_khz[squelch_mask] = 0.0

                channel_offsets_ch = np.array([])
                channel_power_dbm = []
                avg_pwr = 0.0
                peak_pwr = 0.0
                papr = 0.0
                leakage_pwr = -100.0
                df1_avg = 0.0
                df2_avg = 0.0
                df1_max = 0.0
                df2_min = 0.0
                mod_index = 0.0
                freq_drift_khz = 0.0
                
                if not self.skip_metrics:
                    # Espectro ACP (Adjacent Channel Power)
                    N_b = len(burst_samples)
                    # Padear a potencia de 2 para FFT más rápida
                    nfft = 1
                    while nfft < N_b:
                        nfft *= 2
                    fft_vals = np.fft.fftshift(np.fft.fft(burst_samples, n=nfft)) / N_b
                    power_spectrum_b = np.abs(fft_vals) ** 2
    
                    channel_bw = self.bw_mhz * 1e6
                    offsets_mhz = np.arange(-10, 11)
                    channel_offsets_ch = offsets_mhz / 2.0

                    # Cálculo vectorizado: convertir offsets a bins de frecuencia
                    # y sumar potencia por slicing directo en vez de np.where por canal
                    freq_resolution = self.sample_rate / nfft
                    half_bw_bins = int(channel_bw / 2 / freq_resolution)
                    for offset_mhz in offsets_mhz:
                        center_bin = int(offset_mhz * 1e6 / freq_resolution) + nfft // 2
                        lo = max(0, center_bin - half_bw_bins)
                        hi = min(nfft, center_bin + half_bw_bins + 1)
                        if hi > lo:
                            pwr = np.sum(power_spectrum_b[lo:hi])
                            pwr_dbm = 10 * np.log10(pwr + 1e-12)
                        else:
                            pwr_dbm = -100
                        channel_power_dbm.append(float(pwr_dbm))
    
                    # Calcular métricas de potencia (sobre la parte activa de la ráfaga)
                    peak_pwr = float(peak_pwr_dbm)
                    active_mask = power_dbm > (peak_pwr - self._SQUELCH_THRESHOLD_DB)
                    if np.any(active_mask):
                        active_power_mw = power_mw[active_mask]
                        avg_pwr = float(10 * np.log10(np.mean(active_power_mw) + 1e-12))
                    else:
                        avg_pwr = peak_pwr
                    papr = peak_pwr - avg_pwr
                    
                    # Leakage Power: Calculado a partir de los márgenes de silencio de la ráfaga extraída.
                    # Esto evita que otras ráfagas en el buffer grande rompan la medición.
                    idle_mask = power_dbm < (peak_pwr - self._LEAKAGE_THRESHOLD_DB)
                    
                    if np.any(idle_mask):
                        leakage_pwr = float(10 * np.log10(np.mean(power_mw[idle_mask]) + 1e-12))
                    else:
                        leakage_pwr = -100.0

                    # ── Métricas de desviación de frecuencia (TRM-LE/CA/BV-01 a BV-06) ──
                    # Anclar el timing de bits al preámbulo para muestrear en el
                    # centro exacto de cada periodo de bit, evitando las transiciones GFSK.
                    df1_avg = 0.0   # Desviación promedio de bits "1" (positiva)
                    df2_avg = 0.0   # Desviación promedio de bits "0" (negativa)
                    df1_max = 0.0   # Desviación máxima positiva
                    df2_min = 0.0   # Desviación máxima negativa (más negativa)
                    mod_index = 0.0 # Índice de modulación medido
                    freq_drift_khz = 0.0  # Deriva de frecuencia durante el paquete

                    sps = self._samples_per_bit
                    # Ventana de promediado: ±20% del periodo de bit alrededor del centro.
                    # Suficiente para rechazar ruido sin pisar las transiciones GFSK.
                    avg_margin = max(1, sps // 5)
                    
                    if preamble_found and preamble_start is not None:
                        # Offset del preámbulo dentro de la ventana extraída
                        preamble_offset = preamble_start - extract_start
                        # Generar puntos de decisión alineados al timing del preámbulo
                        # El primer bit empieza en preamble_offset, su centro está en +sps//2
                        first_center = preamble_offset + sps // 2
                        centers = np.arange(first_center, min_len, sps)
                    else:
                        # Fallback: empezar desde el inicio de la región activa
                        active_indices = np.where(~squelch_mask)[0]
                        if len(active_indices) > 0:
                            first_center = active_indices[0] + sps // 2
                            centers = np.arange(first_center, min_len, sps)
                        else:
                            centers = np.array([], dtype=int)

                    # Filtrar centros que caigan fuera de rango o en zona squelched
                    valid_centers = centers[(centers >= avg_margin) & 
                                           (centers + avg_margin < min_len) &
                                           (~squelch_mask[centers.astype(int)])]
                    
                    if len(valid_centers) > 0:
                        # Promediar una ventana alrededor de cada centro de bit
                        bit_devs = np.array([
                            np.mean(freq_dev_khz[c - avg_margin:c + avg_margin])
                            for c in valid_centers.astype(int)
                        ])

                        # Separar bits "1" (desviación positiva) y "0" (negativa)
                        pos_devs = bit_devs[bit_devs > 0]
                        neg_devs = bit_devs[bit_devs < 0]

                        if len(pos_devs) > 0:
                            df1_avg = float(np.mean(pos_devs))
                            df1_max = float(np.max(pos_devs))
                        if len(neg_devs) > 0:
                            df2_avg = float(np.mean(neg_devs))
                            df2_min = float(np.min(neg_devs))

                        # Índice de modulación: h = (Δf1avg - Δf2avg) / (bit_rate en kHz)
                        if len(pos_devs) > 0 and len(neg_devs) > 0:
                            mod_index = (df1_avg - df2_avg) / (self.bit_rate / 1000.0)

                        # Frequency Drift: diferencia entre la media de los puntos
                        # de decisión en la primera mitad vs la segunda mitad
                        half = len(bit_devs) // 2
                        if half > 0:
                            first_half_mean = float(np.mean(bit_devs[:half]))
                            second_half_mean = float(np.mean(bit_devs[half:]))
                            freq_drift_khz = second_half_mean - first_half_mean

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
                    # ── Métricas de Desviación de Frecuencia ──
                    'df1_avg_khz': df1_avg,
                    'df2_avg_khz': df2_avg,
                    'df1_max_khz': df1_max,
                    'df2_min_khz': df2_min,
                    'mod_index': mod_index,
                    'freq_drift_khz': freq_drift_khz,
                    'skip_metrics': self.skip_metrics
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

    def _fallback_power_detection(self, iq_samples, freq_dev_hz,
                                  burst_starts, burst_ends):
        """
        Método de respaldo para detectar la región de interés cuando
        no se encuentra un preámbulo BLE válido por correlación.

        Reutiliza los resultados de detección de ráfagas ya calculados
        en el paso principal. Distingue entre:
        - Señal continua (sin bursts detectados): ventana centrada fija
        - Señal con bursts: usa el primer burst válido

        En ambos casos estima CFO como la media de freq_dev en la
        región detectada (asumiendo datos balanceados).

        Args:
            iq_samples: Muestras IQ del buffer
            freq_dev_hz: Desviación de frecuencia ya calculada
            burst_starts: Array de índices de inicio de ráfagas (de _detect_bursts)
            burst_ends: Array de índices de fin de ráfagas (de _detect_bursts)

        Returns:
            tuple: (extract_start, extract_end, cfo_hz)
                   extract_start puede ser None si no se detecta nada
        """
        if len(burst_starts) == 0:
            # Señal continua o sin bursts: ventana centrada de tamaño fijo
            max_display = int(self.sample_rate * self._DISPLAY_WINDOW_S)
            center = len(iq_samples) // 2
            half = min(max_display // 2, center)
            extract_start = center - half
            extract_end = center + half

            # CFO de señal continua (datos balanceados → media ≈ 0)
            cfo_hz = float(np.mean(
                freq_dev_hz[extract_start:extract_end]))
            return extract_start, extract_end, cfo_hz
        else:
            # Señal con bursts: usar el primer burst válido
            s = burst_starts[0]
            e = burst_ends[0]
            margin = int(self.sample_rate * 50e-6)
            extract_start = max(0, s - margin)
            extract_end = min(len(iq_samples), e + margin)

            # CFO del burst (datos balanceados → media ≈ 0)
            cfo_hz = float(np.mean(
                freq_dev_hz[extract_start:extract_end]))
            return extract_start, extract_end, cfo_hz
