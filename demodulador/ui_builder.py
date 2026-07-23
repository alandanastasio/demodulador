from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QPainterPath, QActionGroup
from PyQt6.QtWidgets import QWidget, QStackedWidget, QHBoxLayout, QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QFormLayout, QToolBar, QToolButton, QMenu, QPushButton, QGridLayout, QCheckBox, QFrame
import pyqtgraph as pg
import numpy as np
from marker_manager import MarkerManager

def build_ui(self, state):
    # --- BARRA SUPERIOR  ---
    self.toolbar = QToolBar("Barra Principal")
    self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
    self.toolbar.setMovable(False)

    # 1 Rec/Play
    self.rec_play_btn = QToolButton()
    self.rec_play_btn.setText("Rec/Play")
    self.rec_play_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    self.rec_play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    self.rec_play_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")

    # Botón Pausa/Reanudar
    self.pause_btn = QToolButton()
    self.pause_btn.setText("⏸")
    self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    self.pause_btn.setCheckable(True)
    self.pause_btn.setFixedSize(QSize(45, 40))
    self.pause_btn.setStyleSheet("background-color: #444; color: white; font-size: 16px; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")
    self.pause_btn.clicked.connect(self.toggle_pause)
    self.toolbar.addWidget(self.pause_btn)

    # 2. Crear el Menú que va a contener las opciones
    self.rec_play_menu = QMenu()
    self.rec_play_menu.setStyleSheet("""
        QMenu {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #444;
        }
        QMenu::item:selected {
            background-color: #555555;
        }
    """)

    # 3. Crear las acciones (Opciones del menú)
    self.record_action = QAction("🔴 Iniciar Grabación", self)
    self.record_action.triggered.connect(self.playback_manager.toggle_recording)
    
    self.play_action = QAction(" ▶ Reproducir Archivo", self)
    self.play_action.triggered.connect(lambda: self.playback_manager.load_and_play(loop=False))
    
    # Boton de reproducir en loop y de detener
    self.loop_action = QAction("🔁 Reproducir archivo en loop", self)
    self.loop_action.triggered.connect(lambda: self.playback_manager.load_and_play(loop=True))

    self.stop_play_action = QAction("⏹ Detener Reproducción", self)
    self.stop_play_action.triggered.connect(self.playback_manager.stop_playback)
    self.stop_play_action.setEnabled(False) # Arranca deshabilitado
    
    # 4. Agregar al menú
    self.rec_play_menu.addAction(self.record_action)
    self.rec_play_menu.addAction(self.play_action)
    self.rec_play_menu.addAction(self.loop_action)
    self.rec_play_menu.addSeparator() # Una rayita separadora queda linda
    self.rec_play_menu.addAction(self.stop_play_action)

    self.rec_play_btn.setMenu(self.rec_play_menu)
    self.toolbar.addWidget(self.rec_play_btn)

    self.toolbar.addSeparator() # Una barrita vertical para separar

    main_layout = QHBoxLayout()

    # --- LADO IZQUIERDO: GRÁFICO ---
    self.freq_plot = pg.PlotWidget(labels={'left': 'Potencia [dB]', 'bottom': 'Frecuencia [MHz]'})
    self.freq_plot.setMouseEnabled(x=True, y=True)
    self.freq_plot.setYRange(-130, 10)
    self.freq_plot_curve = self.freq_plot.plot([], pen=pg.mkPen(color='#FFD500', width=1.5))

        # --- QSTACKEDWIDGET PARA MODOS ---
    self.modes_stack = QStackedWidget()

    # ==========================================
    # PÁGINA 0: MODO NORMAL (Waterfall)
    # ==========================================
    self.page_normal = QWidget()
    # Usaremos QGridLayout para poder poner freq_plot arriba y waterfall abajo
    self.layout_normal = QGridLayout(self.page_normal)
    self.layout_normal.setContentsMargins(0, 0, 0, 0)
    self.layout_normal.setSpacing(5)
    
    self.waterfall_widget = pg.PlotWidget(title="Espectrograma (Waterfall)")
    self.waterfall_widget.setLabel('left', 'Tiempo [s]')
    self.waterfall_widget.getViewBox().invertY(True)
    self.waterfall_widget.setXLink(self.freq_plot)
    self.waterfall_image = pg.ImageItem()
    
    # El usuario prefiere el colormap 'turbo'
    self.waterfall_colormap = pg.colormap.get('turbo')
    # Aumentamos la resolución a 1024 colores para tener mayor sensibilidad a cambios finos de potencia
    self.waterfall_image.setLookupTable(self.waterfall_colormap.getLookupTable(nPts=1024))
    self.waterfall_widget.addItem(self.waterfall_image)
    self.waterfall_enabled = False
    self.waterfall_buffer = None
    self.waterfall_lines = 200
    
    self.layout_normal.addWidget(self.waterfall_widget)
    self.modes_stack.addWidget(self.page_normal)

    # ==========================================
    # PÁGINA 1: MODO WBFM
    # ==========================================
    self.page_wbfm = QWidget()
    self.layout_wbfm = QGridLayout(self.page_wbfm)
    self.layout_wbfm.setContentsMargins(0, 0, 0, 0)
    
    self.wbfm_mpx_widget = pg.PlotWidget(title="Espectro MPX (Audio Demodulado)")
    self.wbfm_mpx_widget.setLabel('bottom', 'Frecuencia [kHz]')
    self.wbfm_mpx_widget.setLabel('left', 'Magnitud [dB]')
    self.wbfm_mpx_widget.setXRange(0, 100)
    self.wbfm_mpx_widget.setYRange(-80, 20)
    self.wbfm_mpx_curve = self.wbfm_mpx_widget.plot([], pen=pg.mkPen(color="#C3FF00", width=1.5))
    
    self.wbfm_audio_widget = pg.PlotWidget(title="Señal Demodulada en el Tiempo")
    self.wbfm_audio_widget.setLabel('bottom', 'Tiempo [ms]')
    self.wbfm_audio_widget.setLabel('left', 'Desviación [kHz]') 
    self.wbfm_audio_widget.setXRange(0, 10) 
    self.wbfm_audio_widget.setYRange(-100, 100)
    self.wbfm_audio_curve = self.wbfm_audio_widget.plot([], pen=pg.mkPen(color="#FF9500", width=1.5))
    
    self.wbfm_lr_container = QWidget()
    self.wbfm_lr_layout = QVBoxLayout(self.wbfm_lr_container)
    self.wbfm_lr_layout.setContentsMargins(0,0,0,0)
    self.wbfm_l_widget = pg.PlotWidget(title="Canal Izquierdo (L)")
    self.wbfm_l_widget.setLabel('left', 'Amplitud')
    self.wbfm_l_widget.setXRange(0, 10)
    self.wbfm_l_widget.setYRange(-1.5, 1.5)
    self.wbfm_l_curve = self.wbfm_l_widget.plot([], pen=pg.mkPen(color="#00FFFF", width=1.5))
    
    self.wbfm_r_widget = pg.PlotWidget(title="Canal Derecho (R)")
    self.wbfm_r_widget.setLabel('bottom', 'Tiempo [ms]')
    self.wbfm_r_widget.setLabel('left', 'Amplitud')
    self.wbfm_r_widget.setXRange(0, 10)
    self.wbfm_r_widget.setYRange(-1.5, 1.5)
    self.wbfm_r_curve = self.wbfm_r_widget.plot([], pen=pg.mkPen(color="#FF00FF", width=1.5))
    
    self.wbfm_lr_layout.addWidget(self.wbfm_l_widget)
    self.wbfm_lr_layout.addWidget(self.wbfm_r_widget)
    
    self.layout_wbfm.addWidget(self.wbfm_mpx_widget, 0, 1)
    self.layout_wbfm.addWidget(self.wbfm_audio_widget, 1, 0)
    self.layout_wbfm.addWidget(self.wbfm_lr_container, 1, 1)
    
    self.modes_stack.addWidget(self.page_wbfm)

    # ==========================================
    # PÁGINA 2: MODO WIFI
    # ==========================================
    self.page_wifi = QWidget()
    self.layout_wifi = QGridLayout(self.page_wifi)
    self.layout_wifi.setContentsMargins(0, 0, 0, 0)
    
    self.wifi_time_widget = pg.PlotWidget(title="Señal Baseband en el Tiempo")
    self.wifi_time_widget.setLabel('bottom', 'Tiempo [us]')
    self.wifi_time_widget.setLabel('left', 'Amplitud')
    self.wifi_time_widget.setXRange(0, 350) 
    self.wifi_time_widget.setYRange(0, 1)
    self.wifi_time_curve = self.wifi_time_widget.plot([], pen=pg.mkPen(color="#C3FF00", width=1.5))
    
    self.wifi_evm_subc_widget = pg.PlotWidget(title="EVM por Subportadora")
    self.wifi_evm_subc_widget.setLabel('bottom', 'Subportadora')
    self.wifi_evm_subc_widget.setLabel('left', 'EVM [dB]')
    self.wifi_evm_subc_widget.setXRange(-27, 27)
    self.wifi_evm_subc_widget.setYRange(-40, 0)
    
    self.q3_evm_peak_subc = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush(100, 100, 255, 100))
    self.q3_evm_rms_subc = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush(0, 0, 150, 200))
    self.q3_evm_limit = pg.InfiniteLine(pos=-25, angle=0, pen=pg.mkPen(color="#FFFFFF", style=Qt.PenStyle.DashLine))
    self.wifi_evm_subc_widget.addItem(self.q3_evm_peak_subc)
    self.wifi_evm_subc_widget.addItem(self.q3_evm_rms_subc)
    self.wifi_evm_subc_widget.addItem(self.q3_evm_limit)
    
    self.wifi_evm_sym_widget = pg.PlotWidget(title="EVM por Símbolo")
    self.wifi_evm_sym_widget.setLabel('bottom', 'Símbolo')
    self.wifi_evm_sym_widget.setLabel('left', 'EVM [dB]')
    self.wifi_evm_sym_widget.setYRange(-40, 0)
    
    self.q3_evm_rms_sym = self.wifi_evm_sym_widget.plot([], pen=pg.mkPen(color="#00FF00", width=2))
    self.q3_evm_peak_sym = self.wifi_evm_sym_widget.plot([], pen=pg.mkPen(color="#FF3333", width=1.5, style=Qt.PenStyle.DashLine))
    self.q3b_evm_limit = pg.InfiniteLine(pos=-25, angle=0, pen=pg.mkPen(color="#FFFFFF", style=Qt.PenStyle.DashLine))
    self.wifi_evm_sym_widget.addItem(self.q3_evm_rms_sym)
    self.wifi_evm_sym_widget.addItem(self.q3b_evm_limit)
    
    self.wifi_const_widget = pg.PlotWidget(title="Constelación")
    self.wifi_const_widget.setLabel('bottom', 'En Fase (I)')
    self.wifi_const_widget.setLabel('left', 'Cuadratura (Q)')
    self.wifi_const_widget.setXRange(-1.5, 1.5)
    self.wifi_const_widget.setYRange(-1, 1)
    self.wifi_const_widget.showGrid(x=False, y=False)
    self.wifi_const_widget.setAspectLocked(True)
    
    self.wifi_const_curve = self.wifi_const_widget.plot([], pen=None, symbol='o', symbolSize=5, symbolPen=None, symbolBrush="#00FFFF")
    self.wifi_const_signal_curve = self.wifi_const_widget.plot([], pen=None, symbol='o', symbolSize=2.5, symbolPen="#FF00FF")
    
    cross_path = QPainterPath()
    cross_path.moveTo(-0.5, 0); cross_path.lineTo(0.5, 0)
    cross_path.moveTo(0, -0.5); cross_path.lineTo(0, 0.5)
    self.wifi_const_ideal_curve = self.wifi_const_widget.plot([], pen=None, symbol=cross_path, symbolSize=30, symbolPen=pg.mkPen(color="#606060", width=1), symbolBrush=None)
    self.wifi_const_ideal_curve.hide()
    
    self.btn_ideal_const = QPushButton("⌖", self.wifi_const_widget)
    self.btn_ideal_const.setStyleSheet("QPushButton { background-color: rgba(60, 60, 60, 200); color: white; border-radius: 10px; font-weight: bold; font-size: 16px; border: none; } QPushButton:checked { background-color: rgba(200, 200, 200, 220); color: black; }")
    self.btn_ideal_const.setCheckable(True)
    self.btn_ideal_const.resize(24, 24)
    self.btn_ideal_const.move(10, 10)
    self.btn_ideal_const.setToolTip("Mostrar Constelación Ideal")
    
    # --- ICONOS DE AYUDA (EVM) ---
    self.help_q3 = QLabel("?", self.wifi_evm_subc_widget)
    self.help_q3.setStyleSheet("background-color: rgba(60, 60, 60, 200); color: white; border-radius: 10px; font-weight: bold;")
    self.help_q3.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.help_q3.resize(20, 20)
    self.help_q3.move(10, 10)
    
    self.tooltip_q3 = QLabel("Azul oscuro: EVM RMS de la subportadora\nCeleste: EVM Pico de la subportadora\nVioleta: Portadoras\nBlanco: Límite del estándar", self.wifi_evm_subc_widget)
    self.tooltip_q3.setStyleSheet("background-color: rgba(30, 30, 30, 240); color: white; border: 1px solid #777; padding: 8px; border-radius: 5px; font-size: 14px;")
    self.tooltip_q3.adjustSize()
    self.tooltip_q3.move(35, 10)
    self.tooltip_q3.hide()
    
    self.help_q3.enterEvent = lambda e: self.tooltip_q3.show()
    self.help_q3.leaveEvent = lambda e: self.tooltip_q3.hide()

    self.help_q3b = QLabel("?", self.wifi_evm_sym_widget)
    self.help_q3b.setStyleSheet("background-color: rgba(60, 60, 60, 200); color: white; border-radius: 10px; font-weight: bold;")
    self.help_q3b.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.help_q3b.resize(20, 20)
    self.help_q3b.move(10, 10)
    
    self.tooltip_q3b = QLabel("Verde: EVM RMS de cada símbolo OFDM\nRojo punteado: EVM Pico de cada símbolo\nBlanco: Límite del estándar", self.wifi_evm_sym_widget)
    self.tooltip_q3b.setStyleSheet("background-color: rgba(30, 30, 30, 240); color: white; border: 1px solid #777; padding: 8px; border-radius: 5px; font-size: 14px;")
    self.tooltip_q3b.adjustSize()
    self.tooltip_q3b.move(35, 10)
    self.tooltip_q3b.hide()
    
    self.help_q3b.enterEvent = lambda e: self.tooltip_q3b.show()
    self.help_q3b.leaveEvent = lambda e: self.tooltip_q3b.hide()
    
    self.layout_wifi.addWidget(self.wifi_time_widget, 0, 1)
    self.layout_wifi.addWidget(self.wifi_const_widget, 1, 1, 2, 1)
    self.layout_wifi.addWidget(self.wifi_evm_subc_widget, 1, 0)
    self.layout_wifi.addWidget(self.wifi_evm_sym_widget, 2, 0)
    
    self.modes_stack.addWidget(self.page_wifi)

    # --- CONTENEDOR PRINCIPAL ---
    self.plot_container = QWidget()
    self.plot_layout = QVBoxLayout(self.plot_container)
    self.plot_layout.setContentsMargins(0, 0, 0, 0)
    self.plot_layout.setSpacing(0) 
    self.plot_layout.addWidget(self.modes_stack)
    
    # --- SISTEMA DE MARKERS ---
    self.marker_manager = MarkerManager(self, state['center_freq']/1e6)
    self.marker_manager.attach_to_plots()
    
    # --- Instalamos event filters para doble-click maximizar ---
    self.all_panels = [self.freq_plot, self.waterfall_widget, self.wbfm_mpx_widget, self.wbfm_audio_widget, self.wbfm_lr_container, self.wifi_time_widget, self.wifi_evm_subc_widget, self.wifi_evm_sym_widget, self.wifi_const_widget]
    for w in self.all_panels:
        w.installEventFilter(self)

    # Agregamos el contenedor entero al layout principal
    main_layout.addWidget(self.plot_container, stretch=4)
    # --- MENÚ DE MARKERS ---
    self.markers_btn = QToolButton()
    self.markers_btn.setText("Markers")
    self.markers_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    self.markers_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    self.markers_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")

    self.markers_menu = QMenu()
    self.markers_menu.setStyleSheet("""
        QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
        QMenu::item:selected { background-color: #555555; }
    """)

    # Grupo de acciones (Radio buttons para elegir qué marker mover)
    self.marker_group = QActionGroup(self)
    self.marker_group.setExclusive(True)

    def create_marker_action(text, key):
        action = QAction(text, self)
        action.setCheckable(True)
        action.triggered.connect(lambda checked, k=key: self.marker_manager.select_marker(k))
        self.marker_group.addAction(action)
        self.markers_menu.addAction(action)
        return action

    self.action_m1 = create_marker_action("📍 Seleccionar M1", 'M1')
    self.action_d1 = create_marker_action("📍 Seleccionar Delta 1", 'D1')
    self.action_m2 = create_marker_action("📍 Seleccionar M2", 'M2')
    self.action_d2 = create_marker_action("📍 Seleccionar Delta 2", 'D2')

    self.markers_menu.addSeparator()
    
    self.action_none = QAction("🚫 Mover Ninguno", self)
    self.action_none.setCheckable(True)
    self.action_none.setChecked(True) # Por defecto no se mueve ninguno
    self.action_none.triggered.connect(lambda: self.marker_manager.select_marker(None))
    self.marker_group.addAction(self.action_none)
    self.markers_menu.addAction(self.action_none)

    self.markers_menu.addSeparator()
    
    # --- SUBMENÚ DE ELIMINACIÓN ---
    self.delete_menu = QMenu("🗑️ Eliminar...", self)
    self.delete_menu.setStyleSheet("""
        QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
        QMenu::item:selected { background-color: #8b0000; } /* Un rojito oscuro al seleccionar */
    """)

    def create_delete_action(text, key):
        action = QAction(text, self)
        action.triggered.connect(lambda checked, k=key: self.marker_manager.delete_marker(k))
        self.delete_menu.addAction(action)
        return action

    create_delete_action("❌ Eliminar M1", 'M1')
    create_delete_action("❌ Eliminar Delta 1", 'D1')
    create_delete_action("❌ Eliminar M2", 'M2')
    create_delete_action("❌ Eliminar Delta 2", 'D2')
    
    self.delete_menu.addSeparator()
    
    self.clear_markers_action = QAction("💥 Limpiar Todos", self)
    self.clear_markers_action.triggered.connect(self.marker_manager.clear_markers)
    self.delete_menu.addAction(self.clear_markers_action)

    # Agregar el submenú al menú principal
    self.markers_menu.addMenu(self.delete_menu)

    self.markers_btn.setMenu(self.markers_menu)
    self.toolbar.addWidget(self.markers_btn)


    # --- MENÚ DE DEMODULACIONES ---
    self.demod_btn = QToolButton()
    self.demod_btn.setText("Demodulación")
    self.demod_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    self.demod_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    self.demod_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px; margin: 4px;")

    self.demod_menu = QMenu()
    self.demod_menu.setStyleSheet("""
        QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
        QMenu::item:selected { background-color: #555555; }
    """)

    # Grupo de acciones para que solo una demodulación esté activa a la vez en toda la app
    self.demod_group = QActionGroup(self)
    self.demod_group.setExclusive(True)

    # Acción: Sin Demodular (Por defecto)
    self.action_demod_none = QAction("Sin Demodular", self)
    self.action_demod_none.setCheckable(True)
    self.action_demod_none.setChecked(True)
    self.action_demod_none.triggered.connect(self.set_normal_mode)
    self.demod_group.addAction(self.action_demod_none)
    self.demod_menu.addAction(self.action_demod_none)

    self.demod_menu.addSeparator()

    # --- SUBMENÚ: FM ---
    self.fm_menu = QMenu("FM", self)
    self.fm_menu.setStyleSheet("""
        QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
        QMenu::item:selected { background-color: #555555; }
    """)

    # Opciones dentro del submenú FM
    self.action_wbfm = QAction("WBFM (Radio Comercial)", self)
    self.action_wbfm.setCheckable(True)
    self.action_wbfm.triggered.connect(self.set_wbfm_mode)
    self.demod_group.addAction(self.action_wbfm)
    self.fm_menu.addAction(self.action_wbfm)

    self.action_wbfm_audio = QAction("WBFM (Audio en Vivo)", self)
    self.action_wbfm_audio.setCheckable(True)
    self.action_wbfm_audio.triggered.connect(self.set_wbfm_audio_mode) # Usará una función nueva
    self.demod_group.addAction(self.action_wbfm_audio)
    self.fm_menu.addAction(self.action_wbfm_audio)

    self.action_nbfm = QAction("Custom FM", self)
    self.action_nbfm.setCheckable(True)
    # self.action_nbfm.triggered.connect(...)
    self.demod_group.addAction(self.action_nbfm)
    self.fm_menu.addAction(self.action_nbfm)

    # Agregamos el submenú FM al menú principal de Demodulación
    self.demod_menu.addMenu(self.fm_menu)

    # Asignar menú al botón y agregar a la barra principal
    self.demod_btn.setMenu(self.demod_menu)
    self.toolbar.addWidget(self.demod_btn)

    # --- SUBMENÚ: DIGITAL ---
    self.digital_menu = QMenu("Digitales", self)
    self.digital_menu.setStyleSheet("""
        QMenu { background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; }
        QMenu::item:selected { background-color: #555555; }
    """)

    self.action_wifi_ag = QAction("WiFi 802.11a/g (OFDM)", self)
    self.action_wifi_ag.setCheckable(True)
    self.action_wifi_ag.triggered.connect(self.set_wifi_ag_mode)
    
    self.demod_group.addAction(self.action_wifi_ag)
    self.digital_menu.addAction(self.action_wifi_ag)

    # Agregamos el submenú Digital al menú principal de Demodulación
    self.demod_menu.addMenu(self.digital_menu)

    # --- LADO DERECHO: CONTROLES ---
    controls_layout = QVBoxLayout()
    controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # --- TÍTULO PRINCIPAL ---
    self.sa_config_label = QLabel("CONFIGURACIÓN SA")
    self.sa_config_label.setStyleSheet("color: white; font-weight: bold; font-size: 13px; margin-bottom: 5px;")
    controls_layout.addWidget(self.sa_config_label)

    form_layout = QFormLayout()

   # 1. FRECUENCIA CENTRAL (Común a todos)
    freq_layout = QHBoxLayout() # Layout horizontal para juntar el número y la unidad

    self.freq_input = QDoubleSpinBox()
    self.freq_input.setKeyboardTracking(False)
    self.freq_input.setDecimals(6) # Le damos bastantes decimales para que aguante conversiones
    self.freq_input.setRange(0.0, 6000000000.0) # Rango gigante para cubrir desde Hz a GHz
    
    self.unit_combo = QComboBox()
    self.unit_combo.addItems(["Hz", "kHz", "MHz", "GHz"])
    self.unit_combo.setCurrentText("MHz")
    
    # Agregamos los dos elementos al layout horizontal
    freq_layout.addWidget(self.freq_input)
    freq_layout.addWidget(self.unit_combo)

    # Agregamos el layout compuesto al formulario
    form_layout.addRow(QLabel("FREQ CENTRAL:"), freq_layout)

    # Variables de estado para las unidades
    self.current_freq_multiplier = 1e6
    self.freq_input.setValue(state['center_freq'] / self.current_freq_multiplier)
    self.update_spinbox_step() # Ajusta el salto de las flechitas

    # Conexiones
    self.freq_input.valueChanged.connect(self.on_freq_changed)
    self.unit_combo.currentTextChanged.connect(self.on_unit_changed)

    # 2. SAMPLE RATE (Común, pero con opciones distintas según SDR)
    self.sr_label = QLabel("SAMP RATE:")
    self.sr_combo = QComboBox()
    if "HackRF One" in self.radio.nombre:
        self.sr_combo.addItems(["2 MHz", "4 MHz", "8 MHz", "10 MHz", "12.5 MHz", "16 MHz", "20 MHz"])
        self.sr_combo.setCurrentText("10 MHz")
    elif "RTL-SDR" in self.radio.nombre:
        self.sr_combo.addItems(["1.024 MHz", "2.048 MHz", "2.4 MHz", "2.88 MHz"])
        self.sr_combo.setCurrentText("2.4 MHz")
    elif "Nuand bladeRF x40" in self.radio.nombre:
        self.sr_combo.addItems(["2 MHz", "5 MHz", "10 MHz", "20 MHz", "28 MHz", "40 MHz"])
        self.sr_combo.setCurrentText("20 MHz")
    elif "Ettus USRP B200" in self.radio.nombre:
        # La B200 soporta casi cualquier rate (hasta 56 MHz), ponemos valores enteros seguros
        self.sr_combo.addItems(["2 MHz", "4 MHz", "8 MHz", "10 MHz", "16 MHz", "20 MHz", "32 MHz"])
        # Arrancamos en 2 MHz para evitar el Overflow apenas abre el programa
        self.sr_combo.setCurrentText("2 MHz") 
        
    self.sr_combo.currentTextChanged.connect(self.on_sr_changed)
    form_layout.addRow(self.sr_label, self.sr_combo)

    # 3. GANANCIAS (Aparecen, cambian de nombre o desaparecen)
    self.lna_combo = QComboBox() 
    self.vga_combo = QComboBox()

    if "HackRF One" in self.radio.nombre:
        self.lna_combo.addItems([f"{g} dB" for g in range(0, 48, 8)])
        self.lna_combo.setCurrentText("8 dB")
        self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
        form_layout.addRow(QLabel("LNA GAIN:"), self.lna_combo)

        self.vga_combo.addItems([f"{g} dB" for g in range(0, 64, 2)])
        self.vga_combo.setCurrentText("16 dB")
        self.vga_combo.currentTextChanged.connect(self.on_vga_changed)
        form_layout.addRow(QLabel("VGA GAIN:"), self.vga_combo)

    elif "Nuand bladeRF x40" in self.radio.nombre:
        self.lna_combo.addItems([f"{g} dB" for g in range(0, 61, 5)])
        self.lna_combo.setCurrentText("0 dB") 
        self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
        form_layout.addRow(QLabel("GLOBAL GAIN:"), self.lna_combo)
        
    elif "Ettus USRP B200" in self.radio.nombre:
        # La B200 tiene una ganancia unificada que va de 0 a ~73/76 dB. 
        self.lna_combo.addItems([f"{g} dB" for g in range(0, 76, 5)])
        self.lna_combo.setCurrentText("40 dB") # Arrancamos por la mitad
        self.lna_combo.currentTextChanged.connect(self.on_lna_changed)
        form_layout.addRow(QLabel("RX GAIN:"), self.lna_combo)
        
    # Si es "rtlsdr", directamente NO agregamos los botones de ganancia al layout.

    # 4. FFT y TRACE (Común a todos)
    self.fft_combo = QComboBox()
    self.fft_combo.addItems(["512", "1024", "2048", "4096", "8192"])
    self.fft_combo.setCurrentText("4096")
    self.fft_combo.currentTextChanged.connect(self.on_fft_changed)
    form_layout.addRow(QLabel("TAMAÑO FFT:"), self.fft_combo)

    self.trace_combo = QComboBox()
    self.trace_combo.addItems(["White clear", "Max Hold", "Average"])
    self.trace_combo.setCurrentText("White clear")
    self.trace_combo.currentTextChanged.connect(self.trace_manager.set_mode)
    form_layout.addRow(QLabel("TRACE:"), self.trace_combo)

    # ---  BOTÓN ZERO SPAN ---
    self.zero_span_btn = QPushButton("Spam Cero")
    self.zero_span_btn.setCheckable(True)
    self.zero_span_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    self.zero_span_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #555;")
    self.zero_span_btn.clicked.connect(self.toggle_zero_span)
    self.zero_span_label = QLabel("MODO SA:")
    form_layout.addRow(self.zero_span_label, self.zero_span_btn)

    # Agregamos todo lo del form_layout a la barra lateral
    controls_layout.addLayout(form_layout)

    # --- LÍNEA SEPARADORA ---
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet("background-color: #555;")
    controls_layout.addWidget(line)

    # --- SECCIÓN ESPECTROGRAMA ---
    self.waterfall_label = QLabel("ESPECTROGRAMA")
    self.waterfall_label.setStyleSheet("color: white; font-weight: bold; font-size: 13px; margin-top: 5px;")
    controls_layout.addWidget(self.waterfall_label)

    self.waterfall_checkbox = QCheckBox("Activar Waterfall")
    self.waterfall_checkbox.setStyleSheet("color: #ccc;")
    self.waterfall_checkbox.stateChanged.connect(self.on_waterfall_toggled)

    wf_btns_layout = QHBoxLayout()
    wf_btns_layout.setContentsMargins(0, 0, 0, 0)
    wf_btns_layout.addWidget(self.waterfall_checkbox)
    wf_btns_layout.addStretch()
    
    self.wf_btn_menos = QPushButton("-")
    self.wf_btn_menos.setCursor(Qt.CursorShape.PointingHandCursor)
    self.wf_btn_menos.setFixedSize(QSize(25, 25))
    self.wf_btn_menos.setStyleSheet("background-color: #444; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #555;")
    self.wf_btn_menos.clicked.connect(lambda: self.change_waterfall_lines(-50))
    self.wf_btn_menos.setToolTip("Disminuir tiempo (Líneas)")

    self.wf_lines_label = QLabel(f"{getattr(self, 'waterfall_lines', 200)} líneas")
    self.wf_lines_label.setStyleSheet("color: #aaa; font-size: 14px; font-weight: bold;")
    self.wf_lines_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    self.wf_btn_mas = QPushButton("+")
    self.wf_btn_mas.setCursor(Qt.CursorShape.PointingHandCursor)
    self.wf_btn_mas.setFixedSize(QSize(25, 25))
    self.wf_btn_mas.setStyleSheet("background-color: #444; color: white; font-weight: bold; border-radius: 4px; border: 1px solid #555;")
    self.wf_btn_mas.clicked.connect(lambda: self.change_waterfall_lines(50))
    self.wf_btn_mas.setToolTip("Aumentar tiempo (Líneas)")
    
    wf_btns_layout.addWidget(self.wf_btn_menos)
    wf_btns_layout.addWidget(self.wf_lines_label)
    wf_btns_layout.addWidget(self.wf_btn_mas)

    self.waterfall_controls_widget = QWidget()
    self.waterfall_controls_widget.setLayout(wf_btns_layout)
    controls_layout.addWidget(self.waterfall_controls_widget)

    self.wf_btn_save = QPushButton("💾 Guardar imagen")
    self.wf_btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
    self.wf_btn_save.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #555; margin-top: 5px;")
    self.wf_btn_save.clicked.connect(self.save_waterfall)
    self.wf_btn_save.setToolTip("Guardar Espectrograma como Imagen PNG")
    controls_layout.addWidget(self.wf_btn_save)

    # --- LÍNEA SEPARADORA 2 ---
    line2 = QFrame()
    line2.setFrameShape(QFrame.Shape.HLine)
    line2.setFrameShadow(QFrame.Shadow.Sunken)
    line2.setStyleSheet("background-color: #555;")
    controls_layout.addWidget(line2)
    # 5. BOTONES DE AUDIO ESTÉREO
    self.audio_container = QWidget() # Creamos un contenedor
    audio_layout = QHBoxLayout(self.audio_container)
    audio_layout.setContentsMargins(0, 15, 0, 0)
    
    self.audio_l_btn = QPushButton("🔊 Canal L")
    self.audio_l_btn.setCheckable(True)
    self.audio_l_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    self.audio_l_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")
    self.audio_l_btn.clicked.connect(self.audio_manager.toggle_audio)
    
    self.audio_r_btn = QPushButton("🔊 Canal R")
    self.audio_r_btn.setCheckable(True)
    self.audio_r_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    self.audio_r_btn.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #555;")
    self.audio_r_btn.clicked.connect(self.audio_manager.toggle_audio)
    
    audio_layout.addWidget(self.audio_l_btn)
    audio_layout.addWidget(self.audio_r_btn)
    
    # Agregamos el contenedor al layout principal de controles
    controls_layout.addWidget(self.audio_container)
    
    # Ocultamos el contenedor por defecto al iniciar la app
    self.audio_container.hide() 

    # --- MÉTRICAS FM EN EL PANEL DERECHO ---
    self.fm_metrics_label = QLabel("")
    self.fm_metrics_label.setTextFormat(Qt.TextFormat.RichText)
    self.fm_metrics_label.setStyleSheet("background-color: #1e1e1e; padding: 10px; border-radius: 4px; border: 1px solid #444; margin-top: 15px;")
    self.fm_metrics_label.hide() # Lo ocultamos por defecto
    controls_layout.addWidget(self.fm_metrics_label)

    # --- MÉTRICAS ESTÉREO (SEPARACIÓN) ---
    self.stereo_metrics_label = QLabel("")
    self.stereo_metrics_label.setTextFormat(Qt.TextFormat.RichText)
    self.stereo_metrics_label.setStyleSheet("background-color: #1e1e1e; padding: 10px; border-radius: 4px; border: 1px solid #444; margin-top: 10px;")
    self.stereo_metrics_label.hide()
    controls_layout.addWidget(self.stereo_metrics_label)

    # --- MÉTRICAS WIFI (SIGNAL) ---
    self.wifi_metrics_label = QLabel("")
    self.wifi_metrics_label.setTextFormat(Qt.TextFormat.RichText)
    self.wifi_metrics_label.setStyleSheet("background-color: #1e1e1e; padding: 10px; border-radius: 4px; border: 1px solid #444; margin-top: 10px;")
    self.wifi_metrics_label.hide()
    controls_layout.addWidget(self.wifi_metrics_label)

    # --- MÉTRICAS WIFI HW (CFO/SNR) ---
    self.wifi_hw_metrics_label = QLabel("")
    self.wifi_hw_metrics_label.setTextFormat(Qt.TextFormat.RichText)
    self.wifi_hw_metrics_label.setStyleSheet("background-color: #1e1e1e; padding: 10px; border-radius: 4px; border: 1px solid #444; margin-top: 10px;")
    self.wifi_hw_metrics_label.hide()
    controls_layout.addWidget(self.wifi_hw_metrics_label)
    
    controls_widget = QWidget()
    controls_widget.setLayout(controls_layout)
    controls_widget.setFixedWidth(300)
    main_layout.addWidget(controls_widget, stretch=1)

    central_widget = QWidget()
    central_widget.setLayout(main_layout)
    self.setCentralWidget(central_widget)

