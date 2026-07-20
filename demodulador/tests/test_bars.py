import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import sys

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget()
win.show()

# Create a plot and add a BarGraphItem
plot = win.addPlot()
bg1 = pg.BarGraphItem(x=[1, 2, 3], height=[1, 2, 3], width=0.8, brush='b')
plot.addItem(bg1)

# Try updating the brushes via setOpts
bg1.setOpts(brushes=[pg.mkBrush('r'), pg.mkBrush('g'), pg.mkBrush('b')])

print("Updated opts. Did it work?")
# app.exec() # We can't block here, we just want to see if it runs without error.
