import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
import sys

class Segmento3DViewer:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title="Proyecciones del segmento 3D")
        self.win.resize(1000, 300)

        # Crear 3 plots para XY, XZ, YZ
        self.plot_xy = self.win.addPlot(title="Plano XY")
        self.plot_xy.setLabel('bottom', 'X')
        self.plot_xy.setLabel('left', 'Y')
        self.plot_xy.setAspectLocked(True)
        self.curva_xy = self.plot_xy.plot(pen='r', symbol='o')

        self.win.nextRow()
        self.plot_xz = self.win.addPlot(title="Plano XZ")
        self.plot_xz.setLabel('bottom', 'X')
        self.plot_xz.setLabel('left', 'Z')
        self.plot_xz.setAspectLocked(True)
        self.curva_xz = self.plot_xz.plot(pen='g', symbol='o')

        self.win.nextRow()
        self.plot_yz = self.win.addPlot(title="Plano YZ")
        self.plot_yz.setLabel('bottom', 'Y')
        self.plot_yz.setLabel('left', 'Z')
        self.plot_yz.setAspectLocked(True)
        self.curva_yz = self.plot_yz.plot(pen='b', symbol='o')

        self.win.show()

    def actualizar_segmento(self, p1, p2, longitud):
        p1 = np.array(p1)
        p2 = np.array(p2)
        v = p2 - p1
        norma = np.linalg.norm(v)
        if norma == 0:
            return
        u = v / norma
        pL = p1 + longitud * u

        # Proyecciones
        self.curva_xy.setData([p1[0], pL[0]], [p1[1], pL[1]])
        self.curva_xz.setData([p1[0], pL[0]], [p1[2], pL[2]])
        self.curva_yz.setData([p1[1], pL[1]], [p1[2], pL[2]])

        QtWidgets.QApplication.processEvents()  # Actualiza los gráficos

    def ejecutar(self):
        sys.exit(self.app.exec_())
