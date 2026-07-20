import sys
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtCore import Qt
import pyqtgraph as pg

app = QApplication(sys.argv)
win = pg.PlotWidget()
label = QLabel("?", win)
label.setStyleSheet("background-color: red; color: white; border-radius: 10px; font-weight: bold;")
label.setToolTip("HELLO WORLD")
label.resize(20, 20)
label.move(10, 10)
win.show()

# Quit after 1 sec so the script doesn't hang our agent (we just want to see if it renders/crashes, wait, we can't test tooltips without a mouse).
