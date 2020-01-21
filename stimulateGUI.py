import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 500, 350)
        self.setWindowTitle("Draw text")
        self.show()



if __name__ == "__main__":
    print('1')
