import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

class MarkerManager:
    def __init__(self, main_window, initial_freq_mhz):
        # Guardamos una referencia a la ventana principal para manipularla
        self.ui = main_window 
        self.current_moving_marker = None
        
        # --- ESTADO DE LOS MARKERS ---
        self.markers_info = {
            'M1': {'active': False, 'freq': initial_freq_mhz, 'current_plot': 'rf', 'color': '#00B000', 'item': pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush('#00B000'), symbol='d')},
            'D1': {'active': False, 'freq': initial_freq_mhz, 'current_plot': 'rf', 'color': "#00B000", 'item': pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush("#007E00"), symbol='t')}, 
            'M2': {'active': False, 'freq': initial_freq_mhz, 'current_plot': 'rf', 'color': "#0077FF", 'item': pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush("#0077FF"), symbol='d')},
            'D2': {'active': False, 'freq': initial_freq_mhz, 'current_plot': 'rf', 'color': "#0077FF", 'item': pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush("#0054C2"), symbol='t')}
        }
        
        # Creamos las cajitas de texto
        self.marker_text_box = pg.TextItem(text="", color='#FFFFFF', fill=pg.mkBrush(0, 0, 0, 200), anchor=(1, 0))
        self.mpx_marker_text_box = pg.TextItem(text="", color='#FFFFFF', fill=pg.mkBrush(0, 0, 0, 200), anchor=(1, 0))

    def attach_to_plots(self):
        # Enganchamos todo a la interfaz y le pedimos al auto-rango que los ignore
        self.ui.freq_plot.addItem(self.marker_text_box, ignoreBounds=True)
        self.marker_text_box.hide()
        
        self.ui.wbfm_mpx_widget.addItem(self.mpx_marker_text_box, ignoreBounds=True)
        self.mpx_marker_text_box.hide()

        # Atamos los eventos de clics del mouse usando "self.ui"
        self.ui.freq_plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        self.ui.wbfm_mpx_widget.scene().sigMouseClicked.connect(self.on_mpx_mouse_clicked)

    # === LÓGICA DE CONTROL (CLICS Y MENÚS) ===
    
    def select_marker(self, key):
        self.current_moving_marker = key
        if key is not None:
            if not self.markers_info[key]['active']:
                self.markers_info[key]['active'] = True
                plot_target = self.markers_info[key]['current_plot']
                if plot_target == 'rf':
                    self.ui.freq_plot.addItem(self.markers_info[key]['item'])
                    self.marker_text_box.show()
                else:
                    self.ui.wbfm_mpx_widget.addItem(self.markers_info[key]['item'])
                    self.mpx_marker_text_box.show()

    def clear_markers(self):
        for key, data in self.markers_info.items():
            if data['active']:
                if data['current_plot'] == 'rf':
                    self.ui.freq_plot.removeItem(data['item'])
                else:
                    self.ui.wbfm_mpx_widget.removeItem(data['item'])
            data['active'] = False
        
        self.marker_text_box.hide()
        self.mpx_marker_text_box.hide()
        self.ui.action_none.setChecked(True)
        self.current_moving_marker = None

    def delete_marker(self, key):
        if self.markers_info[key]['active']:
            if self.markers_info[key]['current_plot'] == 'rf':
                self.ui.freq_plot.removeItem(self.markers_info[key]['item'])
            else:
                self.ui.wbfm_mpx_widget.removeItem(self.markers_info[key]['item'])
            self.markers_info[key]['active'] = False
        
        if self.current_moving_marker == key:
            self.ui.action_none.setChecked(True)
            self.current_moving_marker = None
        
        if not any(m['active'] and m['current_plot'] == 'rf' for m in self.markers_info.values()):
            self.marker_text_box.hide()
        if not any(m['active'] and m['current_plot'] == 'mpx' for m in self.markers_info.values()):
            self.mpx_marker_text_box.hide()

    def on_mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.current_moving_marker is not None:
            if self.ui.freq_plot.sceneBoundingRect().contains(event.scenePos()):
                mouse_point = self.ui.freq_plot.getViewBox().mapSceneToView(event.scenePos())
                marker = self.markers_info[self.current_moving_marker]
                marker['freq'] = mouse_point.x()
                
                if marker['current_plot'] != 'rf':
                    self.ui.wbfm_mpx_widget.removeItem(marker['item'])
                    self.ui.freq_plot.addItem(marker['item'])
                    marker['current_plot'] = 'rf'
                    self.marker_text_box.show()
                    if not any(m['active'] and m['current_plot'] == 'mpx' for m in self.markers_info.values()):
                        self.mpx_marker_text_box.hide()

    def on_mpx_mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.current_moving_marker is not None:
            if self.ui.wbfm_mpx_widget.sceneBoundingRect().contains(event.scenePos()):
                mouse_point = self.ui.wbfm_mpx_widget.getViewBox().mapSceneToView(event.scenePos())
                marker = self.markers_info[self.current_moving_marker]
                marker['freq'] = mouse_point.x()
                
                if marker['current_plot'] != 'mpx':
                    self.ui.freq_plot.removeItem(marker['item'])
                    self.ui.wbfm_mpx_widget.addItem(marker['item'])
                    marker['current_plot'] = 'mpx'
                    self.mpx_marker_text_box.show()
                    if not any(m['active'] and m['current_plot'] == 'rf' for m in self.markers_info.values()):
                        self.marker_text_box.hide()

    # === MANEJO DE TECLADO ===

    def handle_key_press(self, event, f_axis, last_f_axis_audio):
        if self.current_moving_marker is None:
            return False # Devolvemos False para avisar que NO lo procesamos
        
        if event.key() not in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            return False

        marker = self.markers_info[self.current_moving_marker]
        
        axis_x = f_axis if marker['current_plot'] == 'rf' else last_f_axis_audio
        
        if axis_x is None or len(axis_x) == 0:
            return True

        current_idx = (np.abs(axis_x - marker['freq'])).argmin()
        step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1

        if event.key() == Qt.Key.Key_Left:
            new_idx = max(0, current_idx - step)
        elif event.key() == Qt.Key.Key_Right:
            new_idx = min(len(axis_x) - 1, current_idx + step)

        marker['freq'] = axis_x[new_idx]
        return True # Devolvemos True porque atrapamos la flechita

    # === LÓGICA DE DIBUJADO (RENDERIZADO) ===

    def _process_single_plot_render(self, plot_name, axis_x, axis_y, text_box_item, plot_widget):
        texto_global = ""
        current_frame_data = {} 
        
        if axis_x is None or axis_y is None or len(axis_x) == 0:
            text_box_item.hide()
            return

        for key, data in self.markers_info.items():
            if data['active'] and data['current_plot'] == plot_name:
                idx = (np.abs(axis_x - data['freq'])).argmin()
                x_val = axis_x[idx]
                y_val = axis_y[idx]
                data['item'].setData([x_val], [y_val])
                current_frame_data[key] = {'x': x_val, 'y': y_val, 'color': data['color']}

        if not current_frame_data:
            text_box_item.hide()
            return

        unit = "MHz" if plot_name == 'rf' else "kHz"

        if 'M1' in current_frame_data:
            m1 = current_frame_data['M1']
            texto_global += f"<span style='color:{m1['color']}'><b>M1:</b> {m1['x']:.3f} {unit} | {m1['y']:.2f} dB</span><br>"
        if 'D1' in current_frame_data:
            d1 = current_frame_data['D1']
            if 'M1' in current_frame_data:
                dx = d1['x'] - m1['x']
                dy = d1['y'] - m1['y']
                texto_global += f"<span style='color:{d1['color']}'><b>Δ1:</b> {dx:+.3f} {unit} | {dy:+.2f} dB</span><br>"
            else:
                texto_global += f"<span style='color:{d1['color']}'><b>Δ1:</b> {d1['x']:.3f} {unit} | {d1['y']:.2f} dB (Falta M1)</span><br>"

        if 'M2' in current_frame_data:
            m2 = current_frame_data['M2']
            texto_global += f"<span style='color:{m2['color']}'><b>M2:</b> {m2['x']:.3f} {unit} | {m2['y']:.2f} dB</span><br>"
        if 'D2' in current_frame_data:
            d2 = current_frame_data['D2']
            if 'M2' in current_frame_data:
                dx = d2['x'] - m2['x']
                dy = d2['y'] - m2['y']
                texto_global += f"<span style='color:{d2['color']}'><b>Δ2:</b> {dx:+.3f} {unit} | {dy:+.2f} dB</span><br>"
            else:
                texto_global += f"<span style='color:{d2['color']}'><b>Δ2:</b> {d2['x']:.3f} {unit} | {d2['y']:.2f} dB (Falta M2)</span><br>"

        text_box_item.setHtml(texto_global)
        text_box_item.show()

        # Leemos la cámara: view_rect[0] es [xmin, xmax], view_rect[1] es [ymin, ymax]
        view_rect = plot_widget.viewRange()
        
        # Lo anclamos exactamente a la esquina superior derecha de la vista
        text_box_item.setPos(view_rect[0][1], view_rect[1][1])

    def update_render(self, display_psd, f_axis, psd_audio, f_axis_audio, demod_mode):
        any_active = any(m['active'] for m in self.markers_info.values())
        if any_active:
            # Renderizar para RF
            self._process_single_plot_render('rf', f_axis, display_psd, self.marker_text_box, self.ui.freq_plot)
            
            # Renderizar para MPX
            if demod_mode in ['wbfm', 'wbfm_audio']:
                self._process_single_plot_render('mpx', f_axis_audio, psd_audio, self.mpx_marker_text_box, self.ui.wbfm_mpx_widget)
        else:
            self.marker_text_box.hide()
            self.mpx_marker_text_box.hide()