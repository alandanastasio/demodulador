import pyqtgraph as pg
bg = pg.BarGraphItem(x=[1], height=[2], width=1)
bg.setOpts(x=[1,2], height=[2,3])
print("setOpts worked!")
