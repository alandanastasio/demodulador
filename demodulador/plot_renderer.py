import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import Qt
import time

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
                    if self.waterfall_buffer is None or self.waterfall_buffer.shape[0] != len(PSD) or self.waterfall_buffer.shape[1] != self.waterfall_lines:
                        self.waterfall_buffer = np.zeros((len(PSD), self.waterfall_lines))
                        self.waterfall_buffer.fill(-130)
                        self.waterfall_counter = 0
                    
                    self.waterfall_counter = getattr(self, 'waterfall_counter', 0) + 1
                    
                    if self.waterfall_counter >= 5:
                        self.waterfall_counter = 0
                        
                        # Desplazamos las columnas de tiempo hacia la derecha (o izquierda, según el waterfall)
                        self.waterfall_buffer = np.roll(self.waterfall_buffer, 1, axis=1)
                        self.waterfall_buffer[:, 0] = display_psd
                        
                        # Actualizamos la imagen (X: frecuencia, Y: tiempo)
                        # autoLevels=False y levels fijos evitan que pyqtgraph colapse calculando
                        # los min/max de la matriz enorme docenas de veces por segundo.
                        # El usuario prefiere escala estática: Mínimo en -100 dBm, máximo en +30 dBm
                        self.waterfall_image.setImage(
                            self.waterfall_buffer, 
                            autoLevels=False, 
                            levels=(-100, 30), 
                            autoDownsample=True
                        )
                        
                        # Ajustamos la escala para que coincida con el eje X de frecuencias
                        f_min, f_max = self.f_axis[0], self.f_axis[-1]
                        
                        # Medimos el tiempo exacto que pasó desde la última vez que agregamos una línea
                        now = time.time()
                        if not hasattr(self, 'wf_last_time'):
                            self.wf_last_time = now
                            dt = 1.0 / 30.0 * 5 # Valor por defecto razonable
                        else:
                            dt = now - self.wf_last_time
                            self.wf_last_time = now
                            
                        # Usamos un filtro pasa-bajos simple para estabilizar la escala
                        if not hasattr(self, 'wf_dt_avg'):
                            self.wf_dt_avg = dt
                        else:
                            self.wf_dt_avg = 0.9 * self.wf_dt_avg + 0.1 * dt
                            
                        # El tiempo total del eje es el tiempo promedio por línea * cantidad de líneas
                        total_time_s = self.wf_dt_avg * self.waterfall_lines
                        
                        self.waterfall_image.setRect(pg.QtCore.QRectF(f_min, 0, f_max - f_min, total_time_s))
                        self.waterfall_widget.setLabel('left', 'Tiempo [s]')

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

        # --- RENDERIZADO ESPECÍFICO DE LTE ---
        if state.get('demod_mode') == 'lte':
            if raw_samples is not None:
                # La señal en 'raw_samples' ya viene decimada (ej. 2000 puntos) representando 10 ms.
                # Ya es la envolvente (valor absoluto), no hace falta np.abs
                t_axis_us = np.linspace(0, 10000, len(raw_samples))
                self.lte_time_curve.setData(t_axis_us, raw_samples)
            
            if evm_data is not None and hasattr(self, 'lte_evm_rms_subc'):
                y_floor = -40
                subc_rms_db = np.asarray(evm_data['subc_rms'])
                subc_peak_db = np.asarray(evm_data['subc_peak'])
                
                brushes_peak = [pg.mkBrush(100, 100, 255, 100) for x in evm_data['subc_x']]
                brushes_rms = [pg.mkBrush(0, 0, 150, 200) for x in evm_data['subc_x']]
                
                self.lte_evm_peak_subc.setOpts(x=evm_data['subc_x'], height=subc_peak_db - y_floor, y0=y_floor, brushes=brushes_peak)
                self.lte_evm_rms_subc.setOpts(x=evm_data['subc_x'], height=subc_rms_db - y_floor, y0=y_floor, brushes=brushes_rms)
                
                n_syms = len(evm_data['sym_rms'])
                if n_syms > 0:
                    sym_x = np.arange(n_syms)
                    self.lte_evm_rms_sym.setData(sym_x, evm_data['sym_rms'])
                    self.lte_evm_peak_sym.setData(sym_x, evm_data['sym_peak'])
                    self.lte_evm_sym_widget.setXRange(0, max(n_syms - 1, 1))
                else:
                    self.lte_evm_rms_sym.setData([], [])
                    self.lte_evm_peak_sym.setData([], [])
            
            if audio_L is not None and audio_R is not None:
                if getattr(self, 'action_show_data', None) and self.action_show_data.isChecked():
                    self.lte_const_curve.setData(
                        audio_L, 
                        audio_R, 
                        pen=None, 
                        symbol='o', 
                        symbolSize=2.5, 
                        symbolPen=None, 
                        symbolBrush="#00FFFF"
                    )
                else:
                    self.lte_const_curve.setData([], [])
                
                if fm_metrics and 'pss_pts' in fm_metrics and 'sss_pts' in fm_metrics:
                    pss_pts = fm_metrics['pss_pts']
                    sss_pts = fm_metrics['sss_pts']
                    pdcch_pts = fm_metrics.get('pdcch_pts', np.array([]))
                    pbch_pts = fm_metrics.get('pbch_pts', np.array([]))
                    crs_pts = fm_metrics.get('crs_pts', np.array([]))
                    pcfich_pts = fm_metrics.get('pcfich_pts', np.array([]))
                    phich_pts = fm_metrics.get('phich_pts', np.array([]))
                    
                    if len(pss_pts) > 0 and getattr(self, 'action_show_pss', None) and self.action_show_pss.isChecked():
                        self.lte_pss_curve.setData(pss_pts.real, pss_pts.imag, pen=None, symbol='o', symbolSize=3.5, symbolPen=None, symbolBrush="#FF6600")
                    else:
                        self.lte_pss_curve.setData([], [])
                        
                    if len(sss_pts) > 0 and getattr(self, 'action_show_sss', None) and self.action_show_sss.isChecked():
                        self.lte_sss_curve.setData(sss_pts.real, sss_pts.imag, pen=None, symbol='o', symbolSize=3.5, symbolPen=None, symbolBrush="#4477FF")
                    else:
                        self.lte_sss_curve.setData([], [])
                        
                    if len(pdcch_pts) > 0 and getattr(self, 'action_show_pdcch', None) and self.action_show_pdcch.isChecked():
                        self.lte_pdcch_curve.setData(pdcch_pts.real, pdcch_pts.imag, pen=None, symbol='o', symbolSize=3.5, symbolPen=None, symbolBrush="#FFFF00")
                    else:
                        self.lte_pdcch_curve.setData([], [])
                        
                    if len(pbch_pts) > 0 and getattr(self, 'action_show_pbch', None) and self.action_show_pbch.isChecked():
                        self.lte_pbch_curve.setData(pbch_pts.real, pbch_pts.imag, pen=None, symbol='o', symbolSize=3.5, symbolPen=None, symbolBrush="#00FF00")
                    else:
                        self.lte_pbch_curve.setData([], [])
                        
                    if len(crs_pts) > 0 and getattr(self, 'action_show_crs', None) and self.action_show_crs.isChecked():
                        self.lte_crs_curve.setData(crs_pts.real, crs_pts.imag, pen=None, symbol='o', symbolSize=3.5, symbolPen=None, symbolBrush="#00AADD")
                    else:
                        self.lte_crs_curve.setData([], [])
                        
                    if len(pcfich_pts) > 0 and getattr(self, 'action_show_pcfich', None) and self.action_show_pcfich.isChecked():
                        self.lte_pcfich_curve.setData(pcfich_pts.real, pcfich_pts.imag, pen=None, symbol='o', symbolSize=3.5, symbolPen=None, symbolBrush="#AA00FF")
                    else:
                        self.lte_pcfich_curve.setData([], [])
                        
                    if len(phich_pts) > 0 and getattr(self, 'action_show_phich', None) and self.action_show_phich.isChecked():
                        self.lte_phich_curve.setData(phich_pts.real, phich_pts.imag, pen=None, symbol='o', symbolSize=3.5, symbolPen=None, symbolBrush="#FF3333")
                    else:
                        self.lte_phich_curve.setData([], [])
                        
                self.lte_const_widget.scene().update()
                
            if fm_metrics and 'lte_metrics' in fm_metrics:
                lte = fm_metrics['lte_metrics']
                if lte is not None:
                    pss_found = lte.get('pss_found', False)
                    trama_valida = lte.get('trama_valida', False)
                    n_id_2 = lte.get('N_id_2', '?')
                    n_id_1 = lte.get('N_id_1', '?')
                    cell_id = lte.get('cell_id', '?')
                    
                    pbch_ok = lte.get('pbch_ok', False)
                    pbch_mib = lte.get('pbch_mib', '')
                    pbch_antenas = lte.get('pbch_antenas', '?')
                    
                    mib_bw = lte.get('mib_bw', '?')
                    mib_phich_dur = lte.get('mib_phich_dur', '?')
                    mib_phich_res = lte.get('mib_phich_res', '?')
                    mib_sfn = lte.get('mib_sfn', '?')
                    
                    pcfich_ok = lte.get('pcfich_ok', False)
                    pcfich_cfi = lte.get('pcfich_cfi', '?')
                    
                    color_pss = "#00FF00" if pss_found else "#FF0000"
                    color_trama = "#00FF00" if trama_valida else "#FF0000"
                    color_pbch = "#00FF00" if pbch_ok else "#FF5555"
                    color_pcfich = "#00FF00" if pcfich_ok else "#FF5555"
                    
                    html_lte = (
                        f"<div style='line-height: 1.5;'>"
                        f"<span style='color: #FFFFFF'><b>PSS Encontrado:</b></span> <span style='color: {color_pss}; font-weight: bold;'>{'SI' if pss_found else 'NO'}</span><br>"
                        f"<span style='color: #FFFFFF'><b>N_ID_2 (Sector):</b></span> <span style='color: #00FFFF; font-weight: bold;'>{n_id_2}</span><br>"
                        f"<span style='color: #FFFFFF'><b>N_ID_1 (Grupo):</b></span> <span style='color: #00FFFF; font-weight: bold;'>{n_id_1}</span><br>"
                        f"<span style='color: #FFFFFF'><b>Cell ID:</b></span> <span style='color: #00FFFF; font-weight: bold;'>{cell_id}</span><br>"
                        f"<span style='color: #FFFFFF'><b>Trama Válida:</b></span> <span style='color: {color_trama}; font-weight: bold;'>{'SI' if trama_valida else 'NO'}</span><br>"
                        f"<span style='color: #FFFFFF'><b>PBCH Decodificado:</b></span> <span style='color: {color_pbch}; font-weight: bold;'>{'OK' if pbch_ok else 'NO'}</span><br>"
                        f"<span style='color: #FFFFFF'><b>Antenas Rx:</b></span> <span style='color: #00FFFF; font-weight: bold;'>{pbch_antenas}</span><br>"
                        f"<span style='color: #FFFFFF'><b>Ancho de Banda:</b></span> <span style='color: #00FF00; font-weight: bold;'>{mib_bw}</span><br>"
                        f"<span style='color: #FFFFFF'><b>PHICH (Dur/Res):</b></span> <span style='color: #00FFFF; font-weight: bold;'>{mib_phich_dur} / {mib_phich_res}</span><br>"
                        f"<span style='color: #FFFFFF'><b>System Frame Num:</b></span> <span style='color: #FFD500; font-weight: bold;'>{mib_sfn}</span><br>"
                        f"<span style='color: #FFFFFF'><b>PCFICH Decodificado:</b></span> <span style='color: {color_pcfich}; font-weight: bold;'>{'OK' if pcfich_ok else 'NO'}</span><br>"
                        f"<span style='color: #FFFFFF'><b>PCFICH CFI:</b></span> <span style='color: #00FFFF; font-weight: bold;'>{pcfich_cfi}</span>"
                        f"</div>"
                    )
                    if hasattr(self, 'lte_metrics_label'):
                        self.lte_metrics_label.setText(html_lte)
                        
                    fs = lte.get('frame_summary', {})
                    if hasattr(self, 'lte_frame_summary') and fs:
                        for row in range(self.lte_frame_summary.rowCount()):
                            item_ch = self.lte_frame_summary.item(row, 0)
                            if item_ch:
                                ch_name = item_ch.text()
                                if ch_name in fs:
                                    evm_str, pwr_str, rb_str = fs[ch_name]
                                    self.lte_frame_summary.item(row, 1).setText(evm_str)
                                    self.lte_frame_summary.item(row, 2).setText(pwr_str)
                                    self.lte_frame_summary.item(row, 4).setText(rb_str)
