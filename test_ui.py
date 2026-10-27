from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
DEMOD_DIR = ROOT_DIR / "demodulador"
if str(DEMOD_DIR) not in sys.path:
    sys.path.insert(0, str(DEMOD_DIR))

from PyQt6.QtWidgets import QApplication
from hardware.mock_sdr_handler import MockSDRHandler
from main import MainWindow


def main():
    app = QApplication(sys.argv)
    radio = MockSDRHandler()
    win = MainWindow(radio)
    win.show()
    print("UI Built Successfully")
    win.close()


if __name__ == "__main__":
    main()

