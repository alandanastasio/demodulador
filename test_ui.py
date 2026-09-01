import sys
from PyQt6.QtWidgets import QApplication
from demodulador.main import MainWindow
from hardware.mock_radio_handler import MockRadioHandler

app = QApplication(sys.argv)
radio = MockRadioHandler()
win = MainWindow(radio)
win.show()
print("UI Built Successfully")
