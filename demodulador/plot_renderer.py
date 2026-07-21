import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import Qt

def render_plot(self, state, PSD, raw_samples, PSD_audio=None, f_axis_audio=None, audio_L=None, audio_R=None, t_axis=None, fm_metrics=None, mpx_time=None, evm_data=None):
    if self.is_paused:
        return
    
    if len(self.f_axis) == len(PSD):
        
        # --- LÓGICA DE TRACE ---
        if state.get('zero_span', False):
            if raw_samples is not None:
                # 1. Calculamos la envolvente de potencia en el tiempo (dB)
                # |I + jQ|^2 nos da la potencia. Sumamos 1e-12 para evitar log(0)
                power_time = 10.0 * np.log10(np.abs(raw_samples)**2 + 1e-12)
                
                # 2. Generamos el eje de tiempo en microsegundos (us)
                t_axis = (np.arange(len(raw_samples)) / state['sample_rate']) * 1e6
                
                # 3. Graficamos la señal temporal
                self.freq_plot_curve.setData(t_axis, power_time)
                
                # Fijamos la ventana temporal para que no baile el gráfico
                self.freq_plot.setXRange(0, t_axis[-1])
                
                # Nota: En Zero Span no llamamos a self.marker_manager.update_render
                # porque los markers están atados a coordenadas de frecuencia.
        else:
            # FLUJO NORMAL (Dominio de Frecuencia)
            if len(self.f_axis) == len(PSD):
                # --- LÓGICA DE TRACE ---
                display_psd = self.trace_manager.process(PSD)
                self.freq_plot_curve.setData(self.f_axis, display_psd)
                
                # --- LÓGICA DE WATERFALL ---
                if getattr(self, 'waterfall_enabled', False) and state.get('demod_mode') == 'none':
                    if self.waterfall_buffer is None or self.waterfall_buffer.shape[0] != len(PSD):
                        self.waterfall_lines = 200
                        self.waterfall_buffer = np.zeros((len(PSD), self.waterfall_lines))
                        self.waterfall_buffer.fill(-130)
                        self.waterfall_counter = 0
                    
                    self.waterfall_counter = getattr(self, 'waterfall_counter', 0) + 1
                    
                    if self.waterfall_counter >= 7:
                        self.waterfall_counter = 0
                        
                        # Desplazamos las columnas de tiempo hacia la derecha (o izquierda, según el waterfall)
                        self.waterfall_buffer = np.roll(self.waterfall_buffer, 1, axis=1)
                        self.waterfall_buffer[:, 0] = display_psd
                        
                        # Actualizamos la imagen (X: frecuencia, Y: tiempo)
                        # autoLevels=False y levels fijos evitan que pyqtgraph colapse calculando
                        # los min/max de la matriz enorme docenas de veces por segundo.
                        self.waterfall_image.setImage(
                            self.waterfall_buffer, 
                            autoLevels=False, 
                            levels=(-130, -30), 
                            autoDownsample=True
                        )
                        
                        # Ajustamos la escala para que coincida con el eje X de frecuencias
                        f_min, f_max = self.f_axis[0], self.f_axis[-1]
                        
                        if raw_samples is not None and state.get('sample_rate'):
                            block_time_ms = (len(raw_samples) / state['sample_rate']) * 1000.0
                            total_time_ms = block_time_ms * 7 * self.waterfall_lines
                        else:
                            total_time_ms = self.waterfall_lines * 7
                            
                        self.waterfall_image.setRect(pg.QtCore.QRectF(f_min, 0, f_max - f_min, total_time_ms))

                # --- LÓGICA DE MARKERS Y DELTAS ---
                self.marker_manager.update_render(
                    display_psd, 
                    self.f_axis, 
                    PSD_audio, 
                    getattr(self, 'last_f_axis_audio', None), 
                    state.get('demod_mode')
                )

        # --- RENDERIZADO ESPECÍFICO DE WIFI A/G ---
        if state.get('demod_mode') == 'wifi_ag':
            if raw_samples is not None:
                # 1. Generamos el eje de tiempo en microsegundos (us)
                t_axis_us = (np.arange(len(raw_samples)) / state['sample_rate']) * 1e6
                
                # 2. Graficamos la envolvente (Magnitud)
                signal_env = np.abs(raw_samples)
                self.wifi_time_curve.setData(t_axis_us, signal_env)
            
            # --- GRÁFICAMOS EVM EN EL Q3 ---
            if evm_data is not None and hasattr(self, 'q3_evm_rms_subc'):
                y_floor = -40
                # Barras crecen hacia arriba desde el piso del gráfico (como en el notebook)
                subc_rms_db = np.asarray(evm_data['subc_rms'])
                subc_peak_db = np.asarray(evm_data['subc_peak'])
                
                # Coloreamos distinto a las subportadoras piloto (-21, -7, 7, 21)
                brushes_peak = [pg.mkBrush(180, 50, 255, 150) if x in [-21, -7, 7, 21] else pg.mkBrush(100, 100, 255, 100) for x in evm_data['subc_x']]
                brushes_rms = [pg.mkBrush(140, 0, 220, 200) if x in [-21, -7, 7, 21] else pg.mkBrush(0, 0, 150, 200) for x in evm_data['subc_x']]
                
                self.q3_evm_peak_subc.setOpts(x=evm_data['subc_x'], height=subc_peak_db - y_floor, y0=y_floor, brushes=brushes_peak)
                self.q3_evm_rms_subc.setOpts(x=evm_data['subc_x'], height=subc_rms_db - y_floor, y0=y_floor, brushes=brushes_rms)
                
                n_syms = len(evm_data['sym_rms'])
                if n_syms > 0:
                    sym_x = np.arange(n_syms)
                    self.q3_evm_rms_sym.setData(sym_x, evm_data['sym_rms'])
                    self.q3_evm_peak_sym.setData(sym_x, evm_data['sym_peak'])
                    self.wifi_evm_sym_widget.setXRange(0, max(n_syms - 1, 1))
                else:
                    self.q3_evm_rms_sym.setData([], [])
                    self.q3_evm_peak_sym.setData([], [])
                    
                metrics = fm_metrics.get('wifi_metrics', {}) if fm_metrics else {}
                mod = metrics.get('mod', '')
                mbps = metrics.get('mbps', '')
                if mod:
                    # Límites EVM del estándar IEEE 802.11a/g (Table 17-10)
                    EVM_LIMITS_DB = {
                        6: -5, 9: -8, 12: -10, 18: -13,
                        24: -16, 36: -19, 48: -22, 54: -25
                    }
                    limite_db = EVM_LIMITS_DB.get(int(mbps), -25)
                    self.q3_evm_limit.setPos(limite_db)
                    self.q3b_evm_limit.setPos(limite_db)
                    self.wifi_evm_subc_widget.setTitle(f"EVM por Subportadora ({mod} / {mbps} Mbps) — Límite: {limite_db} dB")
                    self.wifi_evm_sym_widget.setTitle(f"EVM por Símbolo ({mod} / {mbps} Mbps) — Límite: {limite_db} dB")
            elif mpx_time is not None and not hasattr(self, 'q3_evm_rms_subc'):
                eje_x_sc = np.arange(len(mpx_time))
                self.wifi_evm_subc_curve.setData(eje_x_sc, mpx_time) # legacy

            # --- GRAFICAMOS LAS CONSTELACIÓNES EN Q4 ---
            # Recibimos las coordenadas X(I) e Y(Q) a través de los canales de audio
            if audio_L is not None and audio_R is not None:
                # 1. Pasamos nuevamente los parámetros del símbolo para refrescar el ScatterPlotItem
                self.wifi_const_curve.setData(
                    audio_L, 
                    audio_R, 
                    pen=None, 
                    symbol='o', 
                    symbolSize=2.5, 
                    symbolPen=None, 
                    symbolBrush="#00FFFF"
                )
            if PSD_audio is not None and f_axis_audio is not None:
                self.wifi_const_signal_curve.setData(
                    PSD_audio, 
                    f_axis_audio, 
                    pen=None, 
                    symbol='o', 
                    symbolSize=2.5, 
                    symbolPen="#FF00FF", 
                    symbolBrush=None
                )
                
                # 2. Forzamos a Qt a repintar la escena del widget de forma manual e inmediata
                self.wifi_const_widget.scene().update()
            
            # Actualizar Constelación Ideal si el botón está activado
            mod = fm_metrics.get('wifi_metrics', {}).get('mod', '') if fm_metrics else ''
            if mod and hasattr(self, 'btn_ideal_const') and self.btn_ideal_const.isChecked():
                if mod == 'BPSK': niveles = [-1, 1]
                elif mod == 'QPSK': niveles = [-1, 1]
                elif mod == '16-QAM': niveles = [-3, -1, 1, 3]
                elif mod == '64-QAM': niveles = [-7, -5, -3, -1, 1, 3, 5, 7]
                else: niveles = []
                
                if niveles:
                    import itertools
                    pts = [(x, 0) for x in niveles] if mod == 'BPSK' else list(itertools.product(niveles, niveles))
                    pts_arr = np.array(pts, dtype=float)
                    multiplicador = 1 if mod == 'BPSK' else 2
                    P_teorica = np.mean(np.array(niveles)**2) * multiplicador
                    escala = np.sqrt(1.0 / P_teorica)
                    pts_arr *= escala
                    self.wifi_const_ideal_curve.setData(pts_arr[:,0], pts_arr[:,1])
                    self.wifi_const_ideal_curve.show()
                else:
                    self.wifi_const_ideal_curve.hide()
            else:
                if hasattr(self, 'wifi_const_ideal_curve'):
                    self.wifi_const_ideal_curve.hide()
        

        if state.get('demod_mode') in ['wbfm', 'wbfm_audio']:
            if PSD_audio is not None:
                self.wbfm_mpx_curve.setData(f_axis_audio, PSD_audio)
            if mpx_time is not None:                              
                self.wbfm_audio_curve.setData(t_axis, mpx_time)
            if audio_L is not None and audio_R is not None:
                self.wbfm_l_curve.setData(t_axis, audio_L)
                self.wbfm_r_curve.setData(t_axis, audio_R)
            
            # Renderizar las métricas en HTML en el panel derecho
            if fm_metrics is not None:
                html_text = (
                    f"<div style='line-height: 1.5;'>"
                    f"<span style='color: #FFFFFF'><b>Desv. Pico Máx:</b></span> <span style='color: #00B000;'>{fm_metrics['pico_max']:+.2f} kHz</span><br>"
                    f"<span style='color: #FFFFFF'><b>Desv. Pico Mín:</b></span> <span style='color: #FF3333;'>{fm_metrics['pico_min']:+.2f} kHz</span><br>"
                    f"<span style='color: #FFFFFF'><b>Desv. Pico RMS:</b></span> <span style='color: #FFD500;'>{fm_metrics['pico_rms']:.2f} kHz</span><br>"
                    f"<span style='color: #FFFFFF'><b>RMS (True):</b></span> <span style='color: #0077FF;'>{fm_metrics['rms']:.2f} kHz</span><br>"
                    f"<span style='color: #FFFFFF'><b>DC Offset:</b></span> <span style='color: #00FFFF;'>{fm_metrics['dc_offset']:+.2f} kHz</span>"
                    f"</div>"
                )
                self.fm_metrics_label.setText(html_text)
                
            # ---  Cálculo y renderizado de Separación Estéreo ---
                if 'rms_L' in fm_metrics and 'rms_R' in fm_metrics:
                    # Protegemos el logaritmo por si hay silencio absoluto
                    rms_L_val = max(fm_metrics['rms_L'], 1e-12)
                    rms_R_val = max(fm_metrics['rms_R'], 1e-12)
                    
                    db_L = 20 * np.log10(rms_L_val)
                    db_R = 20 * np.log10(rms_R_val)
                    
                    # La separación es la diferencia absoluta de energía
                    separacion_db = abs(db_L - db_R)
                    
                    # Código de colores (50dB+ es excelente para grado laboratorio)
                    if separacion_db >= 50:
                        color_sep = "#00B000" # Verde
                    elif separacion_db >= 30:
                        color_sep = "#FFD500" # Amarillo
                    else:
                        color_sep = "#FF3333" # Rojo (Música normal o mal filtrado)
                        
                    html_stereo = (
                        f"<div style='line-height: 1.5;'>"
                        f"<span style='color: #FFFFFF'><b>Potencia RMS (L):</b></span> <span style='color: #00FFFF;'>{db_L:+.2f} dB</span><br>"
                        f"<span style='color: #FFFFFF'><b>Potencia RMS (R):</b></span> <span style='color: #FF00FF;'>{db_R:+.2f} dB</span><br>"
                        f"<span style='color: #555555;'>─────────────────────────────</span><br>"
                        f"<span style='color: #FFFFFF'><b>Crosstalk (Separación):</b></span> "
                        f"<span style='color: {color_sep}; font-weight: bold;'>{separacion_db:.2f} dB</span>"
                        f"</div>"
                    )
                    self.stereo_metrics_label.setText(html_stereo)

        if state.get('demod_mode') == 'wifi_ag':
            if fm_metrics and 'wifi_metrics' in fm_metrics:
                wifi = fm_metrics['wifi_metrics']
                if wifi is not None:
                    paridad_ok = wifi.get('paridad_ok', False)
                    color_paridad = "#00B000" if paridad_ok else "#FF3333"
                    texto_paridad = "Ok" if paridad_ok else "ERROR"
                    html_wifi = (
                        f"<div style='line-height: 1.5;'>"
                        f"<span style='color: #FFFFFF'><b>Rate Code:</b></span> <span style='color: #00FFFF;'>{wifi.get('rate_code', '?')}</span><br>"
                        f"<span style='color: #FFFFFF'><b>Modulación:</b></span> <span style='color: #FFD500;'>{wifi.get('mod', '?')}</span><br>"
                        f"<span style='color: #FFFFFF'><b>Code rate:</b></span> <span style='color: #FFD500;'>{wifi.get('code_rate', '?')}</span><br>"
                        f"<span style='color: #FFFFFF'><b>Data rate:</b></span> <span style='color: #00FFFF;'>{wifi.get('mbps', 0)} mbps</span><br>"
                        f"<span style='color: #FFFFFF'><b>Length:</b></span> <span style='color: #00FFFF;'>{wifi.get('length', 0)}</span><br>"
                        f"<span style='color: #FFFFFF'><b>Paridad:</b></span> <span style='color: {color_paridad}; font-weight: bold;'>{texto_paridad}</span>"
                        f"</div>"
                    )
                    self.wifi_metrics_label.setText(html_wifi)
                    
                    cfo = wifi.get('cfo', 0.0)
                    cfo_fino = wifi.get('cfo_fino', 0.0)
                    snr = wifi.get('snr', 0.0)
                    html_hw = (
                        f"<div style='line-height: 1.5;'>"
                        f"<span style='color: #FFFFFF'><b>CFO Grueso:</b></span> <span style='color: #FF5555;'>{cfo:+.1f} Hz</span><br>"
                        f"<span style='color: #FFFFFF'><b>CFO Fino:</b></span> <span style='color: #FF5555;'>{cfo_fino:+.1f} Hz</span><br>"
                        f"<span style='color: #FFFFFF'><b>SNR:</b></span> <span style='color: #55FF55;'>{snr:.1f} dB</span>"
                        f"</div>"
                    )
                    self.wifi_hw_metrics_label.setText(html_hw)
