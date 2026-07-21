import sys
import usb.core
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QPushButton, QListWidget, QWidget, QListWidgetItem, QApplication
from hardware.hackrf_handler import HackRFHandler
from hardware.rtlsdr_handler import RtlSdrHandler
from hardware.nuand_bladerf_handler import BladeRFHandler
from hardware.ettus_usrpb200_handler import USRPB200Handler
from main import MainWindow

class StartupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEMODULADOR SDR")
        self.resize(QSize(400, 350))
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        layout = QVBoxLayout()
        label = QLabel("Dispositivos SDR detectados por USB:")
        label.setStyleSheet("font-size: 17px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(label)

        self.scan_btn = QPushButton("🔄 Escanear Puertos USB")
        self.scan_btn.setCursor(Qt.CursorShape.PointingHandCursor) # Cambia la flecha por la manito
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444; 
                color: white; 
                font-weight: bold; 
                padding: 10px 15px; 
                border-radius: 6px; 
                border: 1px solid #5a5a5a;
            }
            QPushButton:hover {
                background-color: #555555;
                border: 1px solid #777777;
            }
            QPushButton:pressed {
                background-color: #2b2b2b;
                border: 1px solid #222222;
            }
        """)
        self.scan_btn.clicked.connect(self.scan_devices)
        layout.addWidget(self.scan_btn)

        self.device_list = QListWidget()
        self.device_list.setStyleSheet("""
            QListWidget { background-color: #1e1e1e; border: 1px solid #444; font-size: 14px; outline: none; }
            QListWidget::item { padding: 15px; }
            QListWidget::item:selected { background-color: #0077FF; color: white; }
            QListWidget::item:hover { background-color: #444444; }
        """)
        self.device_list.itemClicked.connect(self.launch_main_window)
        layout.addWidget(self.device_list)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.scan_devices()

    def scan_devices(self):
        self.device_list.clear()
        devices_found = 0

        try:
            if usb.core.find(idVendor=0x1d50, idProduct=0x6089):
                self.device_list.addItem("HackRF One")
                devices_found += 1
        except: pass
        try:
            if usb.core.find(idVendor=0x0bda, idProduct=0x2838):
                self.device_list.addItem("RTL-SDR")
                devices_found += 1
        except: pass
        try:
            if usb.core.find(idVendor=0x2cf0) or usb.core.find(idVendor=0x1d50, idProduct=0x6066):
                self.device_list.addItem("Nuand bladeRF x40")
                devices_found += 1
        except: pass
        try:
            if usb.core.find(idVendor=0x2500):
                self.device_list.addItem("Ettus USRP B200")
                devices_found += 1
        except: pass

        if devices_found == 0:
            item = QListWidgetItem("⚠️ No se encontraron dispositivos SDR")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.device_list.addItem(item)

    def launch_main_window(self, item):
        device_name = item.text()
        radio_handler = None
        
        # --- INSTANCIACIÓN MODULAR DEL HARDWARE ---
        if "HackRF" in device_name:
            print("Iniciando HackRF...")
            radio_handler = HackRFHandler(rx_callback=None)
        elif "RTL-SDR" in device_name:
            print("Iniciando RTL-SDR...")
            radio_handler = RtlSdrHandler(rx_callback=None)
        elif "bladeRF" in device_name:
            print("Iniciando bladeRF...")
            radio_handler = BladeRFHandler(rx_callback=None)
        elif "Ettus USRP B200" in device_name:
            print("Iniciando Ettus B200...")
            radio_handler = USRPB200Handler(rx_callback=None)
        else:
            return

        # Pasamos el handler a la ventana principal
        self.main_app_window = MainWindow(radio_handler)
        self.main_app_window.show()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    startup_window = StartupWindow()
    startup_window.show()
    sys.exit(app.exec())
