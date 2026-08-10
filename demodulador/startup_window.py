import sys
import os
import usb.core
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QPushButton, QListWidget, QWidget, QListWidgetItem, QApplication
from hardware.hackrf_handler import HackRFHandler
from hardware.rtlsdr_handler import RtlSdrHandler
from hardware.nuand_bladerf_handler import BladeRFHandler
from hardware.ettus_usrpb200_handler import USRPB200Handler
from main import MainWindow

class DeviceScannerThread(QThread):
    devices_found_signal = pyqtSignal(list)

    def run(self):
        import subprocess
        # Pre-carga de la FPGA usando un proceso externo para no bloquear el GIL de Python
        # y permitir que la interfaz (el hilo principal) siga respondiendo.
        try:
            subprocess.run(["uhd_find_devices"], capture_output=True, text=True, timeout=60)
        except Exception as e:
            print("Error al ejecutar uhd_find_devices en subproceso:", e)
            
        found_devices = []
        try:
            if usb.core.find(idVendor=0x1d50, idProduct=0x6089):
                found_devices.append("HackRF One")
        except: pass
        
        try:
            if usb.core.find(idVendor=0x0bda, idProduct=0x2838):
                found_devices.append("RTL-SDR")
        except: pass
        
        try:
            if usb.core.find(idVendor=0x2cf0) or usb.core.find(idVendor=0x1d50, idProduct=0x6066):
                found_devices.append("Nuand bladeRF x40")
        except: pass
        
        try:
            import uhd
            for dev_addr in uhd.find("type=b200"):
                try:
                    info = dev_addr.to_dict()
                    sn = info.get("serial")
                    if sn:
                        found_devices.append(f"Ettus USRP B200 (SN: {sn})")
                    else:
                        found_devices.append("Ettus USRP B200")
                except:
                    found_devices.append("Ettus USRP B200")
        except: pass
        
        self.devices_found_signal.emit(found_devices)

class DeviceInitializerThread(QThread):
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.radio_handler = None

    def run(self):
        try:
            if "HackRF" in self.device_name:
                print("Iniciando HackRF...")
                self.radio_handler = HackRFHandler(rx_callback=None)
            elif "RTL-SDR" in self.device_name:
                print("Iniciando RTL-SDR...")
                self.radio_handler = RtlSdrHandler(rx_callback=None)
            elif "bladeRF" in self.device_name:
                print("Iniciando bladeRF...")
                self.radio_handler = BladeRFHandler(rx_callback=None)
            elif "Ettus USRP B200" in self.device_name:
                print(f"Iniciando {self.device_name}...")
                serial = None
                if "SN: " in self.device_name:
                    serial = self.device_name.split("SN: ")[1].strip(" )")
                
                # Pre-cargar la FPGA usando uhd_usrp_probe en un subproceso
                # Esto evita que uhd.usrp.MultiUSRP bloquee el GIL de Python y congele la UI
                import subprocess
                probe_args = ["uhd_usrp_probe", "--args", f"type=b200" + (f",serial={serial}" if serial else "")]
                try:
                    print("Ejecutando uhd_usrp_probe para cargar FPGA de forma segura...")
                    subprocess.run(probe_args, capture_output=True, text=True, timeout=90)
                except Exception as e:
                    print("Error al ejecutar uhd_usrp_probe:", e)

                self.radio_handler = USRPB200Handler(rx_callback=None, serial=serial)
            
            self.finished_signal.emit(self.radio_handler)
        except Exception as e:
            self.error_signal.emit(str(e))

class StartupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEMODULADOR SDR")
        icon_path = os.path.join(os.path.dirname(__file__), "logo.demod.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(QSize(400, 380))
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

        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet("color: #E0E0E0; font-size: 15px; font-weight: bold; margin-top: 10px;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        layout.addWidget(self.loading_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Variables para la animación de los puntos
        self.dot_count = 1
        self.base_loading_text = ""
        self.dot_timer = QTimer(self)
        self.dot_timer.timeout.connect(self.update_loading_dots)

        # Usar un QTimer para iniciar el escaneo después de que la ventana se haya mostrado
        QTimer.singleShot(100, self.scan_devices)

    def update_loading_dots(self):
        dots = "." * self.dot_count
        self.loading_label.setText(f"{self.base_loading_text}{dots}")
        self.dot_count = (self.dot_count + 1) % 4

    def scan_devices(self):
        self.device_list.clear()
        self.scan_btn.setEnabled(False)
        self.base_loading_text = "🔎 Buscando dispositivos SDR"
        self.loading_label.setText(self.base_loading_text)
        self.dot_count = 1
        self.dot_timer.start(500)
        self.loading_label.show()
        
        self.scanner_thread = DeviceScannerThread()
        self.scanner_thread.devices_found_signal.connect(self.on_scan_completed)
        self.scanner_thread.start()
        
    def on_scan_completed(self, found_devices):
        self.dot_timer.stop()
        self.device_list.clear()
        self.loading_label.hide()
        self.scan_btn.setEnabled(True)
        
        if not found_devices:
            item = QListWidgetItem("⚠️ No se encontraron dispositivos SDR")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.device_list.addItem(item)
        else:
            for dev in found_devices:
                self.device_list.addItem(dev)

    def launch_main_window(self, item):
        device_name = item.text()
        if "No se encontraron" in device_name: return

        self.scan_btn.setEnabled(False)
        self.device_list.setEnabled(False)
        
        if "Ettus" in device_name:
            self.base_loading_text = "⏳ Cargando imagen FPGA"
        else:
            self.base_loading_text = "⏳ Inicializando dispositivo"
            
        self.loading_label.setText(self.base_loading_text)
        self.dot_count = 1
        self.dot_timer.start(500)
        self.loading_label.show()

        self.thread = DeviceInitializerThread(device_name)
        self.thread.finished_signal.connect(self.on_device_initialized)
        self.thread.error_signal.connect(self.on_device_error)
        self.thread.start()

    def on_device_initialized(self, radio_handler):
        self.dot_timer.stop()
        if radio_handler:
            self.main_app_window = MainWindow(radio_handler)
            self.main_app_window.show()
            self.close()
        else:
            self.on_device_error("Handler no fue devuelto correctamente.")

    def on_device_error(self, err_msg):
        self.dot_timer.stop()
        self.loading_label.setText(f"❌ Error al iniciar: {err_msg}")
        self.scan_btn.setEnabled(True)
        self.device_list.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setDesktopFileName("demodulador.desktop")
    icon_path = os.path.join(os.path.dirname(__file__), "logo.demod.png")
    app.setWindowIcon(QIcon(icon_path))
    startup_window = StartupWindow()
    startup_window.show()
    sys.exit(app.exec())
