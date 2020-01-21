import sys

from PyQt5.QtGui import QPainter, QFont, QColor, QPen
import pyqtgraph.opengl as gl
from animated3D import Visualizer, parameter
from softArm import softArm
from math import pi, degrees, acos, atan2, sqrt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from SoftObject import SoftObject
import serial
import threading
import time

class Example(QWidget):
    def __init__(self, sf):
        super().__init__()
        self.initUI(sf)

    def initUI(self, sf):
        self.myserial = serial.Serial()
        self.flag = 0
        self.backward_flag = 0
        self.pre = [.0] * 6

        grid = QGridLayout()

        self.lab1 = QLabel('0')
        self.lab2 = QLabel('0')
        self.lab3 = QLabel('0')
        self.lab4 = QLabel('0')
        self.lab5 = QLabel('0')
        self.lab6 = QLabel('0')
        self.lab7 = QLabel('0')
        self.lab8 = QLabel('0')
        self.lab9 = QLabel('0')

        grid.addWidget(self.lab1, 0, 0)
        grid.addWidget(self.lab2, 1, 0)
        grid.addWidget(self.lab3, 2, 0)
        grid.addWidget(self.lab4, 0, 1)
        grid.addWidget(self.lab5, 1, 1)
        grid.addWidget(self.lab6, 2, 1)
        grid.addWidget(self.lab7, 0, 2)
        grid.addWidget(self.lab8, 1, 2)
        grid.addWidget(self.lab9, 2, 2)

        self.lab10 = QLabel('x')
        self.lab11 = QLabel('y')
        self.lab12 = QLabel('z')

        grid.addWidget(self.lab10, 3, 0)
        grid.addWidget(self.lab11, 4, 0)
        grid.addWidget(self.lab12, 5, 0)

        self.lab13 = QLabel('nx')
        self.lab14 = QLabel('ny')
        self.lab15 = QLabel('nz')

        grid.addWidget(self.lab13, 3, 2)
        grid.addWidget(self.lab14, 4, 2)
        grid.addWidget(self.lab15, 5, 2)

        self.lab20 = QLabel('P1')
        self.lab21 = QLabel('P2')
        self.lab22 = QLabel('P3')
        self.lab23 = QLabel('P4')
        self.lab24 = QLabel('P5')
        self.lab25 = QLabel('P6')

        grid.addWidget(self.lab20, 0, 4, 1, 2)
        grid.addWidget(self.lab21, 1, 4, 1, 2)
        grid.addWidget(self.lab22, 2, 4, 1, 2)
        grid.addWidget(self.lab23, 3, 4, 1, 2)
        grid.addWidget(self.lab24, 4, 4, 1, 2)
        grid.addWidget(self.lab25, 5, 4, 1, 2)

        self.edit10 = QDoubleSpinBox()  # alpha2 beta2 length2
        self.edit10.setSingleStep(0.1)
        self.edit10.setMinimum(-90)
        self.edit10.setMaximum(90)
        self.edit11 = QDoubleSpinBox()
        self.edit11.setSingleStep(0.1)
        self.edit11.setMaximum(90)
        self.edit11.setMinimum(-90)
        self.edit12 = QDoubleSpinBox()
        self.edit12.setSingleStep(0.1)
        self.edit12.setMaximum(90)
        self.edit12.setMinimum(-90)

        grid.addWidget(self.edit10, 3, 1)
        grid.addWidget(self.edit11, 4, 1)
        grid.addWidget(self.edit12, 5, 1)

        self.edit13 = QDoubleSpinBox()  #
        self.edit13.setSingleStep(0.1)
        self.edit13.setMaximum(90)
        self.edit13.setMinimum(-90)
        self.edit14 = QDoubleSpinBox()
        self.edit14.setSingleStep(0.1)
        self.edit14.setMaximum(90)
        self.edit14.setMinimum(-90)
        self.edit15 = QDoubleSpinBox()
        self.edit15.setSingleStep(0.1)
        self.edit15.setMaximum(90)
        self.edit15.setMinimum(-90)

        grid.addWidget(self.edit13, 3, 3)
        grid.addWidget(self.edit14, 4, 3)
        grid.addWidget(self.edit15, 5, 3)

        self.btn1 = QPushButton('serial start',self)
        self.btn2 = QPushButton('change', self)
        self.btn3 = QPushButton('exit', self)
        self.btn4 = QPushButton('display', self)

        grid.addWidget(self.btn1, 0, 6)
        grid.addWidget(self.btn2, 1, 6)
        grid.addWidget(self.btn3, 2, 6)
        grid.addWidget(self.btn4, 3, 6)

        self.view = Visualizer(sf)
        grid.addWidget(self.view.w, 6, 0, 6, 7)

        self.btn1.clicked.connect(self.start)
        self.btn2.clicked.connect(self.changeValue)
        self.btn3.clicked.connect(sys.exit)
        self.btn4.clicked.connect(self.display)
        self.view.Sign.ValueSign.connect(self.Update)

        self.setGeometry(0, 50, 900, 900)
        self.setWindowTitle('feedback')
        self.backup = list()

        self.Update()

        self.setLayout(grid)
        self.show()
        self.view.animation()
        self.edit10.setValue(self.view.pts[8][18][0]/10)
        self.edit11.setValue(-self.view.pts[8][18][1]/10)
        self.edit12.setValue(-self.view.pts[8][18][2]/10)

    def start(self):
        if not self.flag:
            self.serial_enable()
        else:
            self.flag = 1

    def changeValue(self):
        nx = float(self.edit13.text())
        ny = float(self.edit14.text())
        nz = float(self.edit15.text())
        total = sqrt((nx**2+ny**2+nz**2))

        re = self.view.backward_position(float(self.edit10.text()), float(self.edit11.text()), float(self.edit12.text()),
                                    nx/total, ny/total, nz/total)

    def display(self):
        self.view.display = not self.view.display

    def serial_enable(self):
        self.serial()
        try:
            self.flag = 1
            self.receiver_thread = threading.Thread(target=self.reader, name='rec')
            self.receiver_thread.daemon = True
            self.receiver_thread.start()

            self.writer_thread = threading.Thread(target=self.writer, name='wtr')
            self.writer_thread.daemon = True
            self.writer_thread.start()
        except:
            print('初始化失败')

    def serial(self):
        try:
            if self.flag == 0:
                self.myserial = serial.Serial()
                self.myserial.port = 'COM1'
                self.myserial.baudrate = '115200'
                self.myserial.stopbits = 1  # 停止位
                self.myserial.timeout = 0.2
                self.myserial.open()
                print('connnected')

        except serial.SerialException:
            print("reader exception")

    def reader(self):
        print('reader ready')
        while 1:
            if self.flag == 0:
                break
            data = self.myserial.readline()
            if data != b'':
                data = data.decode()
                index = []
                j = 0
                for i in range(6):
                    index.append(data.find('s', j))
                    index.append(data.find('s', j + 1))
                    length = int(data[index[2 * i] + 1:index[2 * i + 1]])
                    j += int(length) + 2 + len(str(length))
                    self.pre[i]=float(data[index[2 * i + 1] + 1:index[2 * i + 1] + length + 1])

    def writer(self):
        print('writer ready')
        while 1:
            if self.flag == 0:
                break
            pre = self.list2byte()

            self.myserial.write(pre)
            self.myserial.flush()
            time.sleep(1)

    def list2byte(self):#将list中的pressure转换为指定格式的bytes 报头 s字符长度s eol \r\n
        pre = self.view.soft.ABLD2PD()
        separate = b's'
        eol = b'\r\n'
        byte_data = bytes()

        for i in range(6):
            lens = str(len(str(pre[i])))
            byte_data += separate + lens.encode() + separate + str(pre[i]).encode()
        byte_data += eol
        return byte_data

    def Update(self):
        self.lab1.setText('alpha1 {0:.2f}'.format(3*degrees(self.view.alpha[0])))
        self.lab2.setText('beta1  {0:.2f}'.format(degrees(self.view.beta[0])))
        self.lab3.setText('length1 {0:.3} m'.format(3*self.view.lm[0]))
        self.lab4.setText('alpha2 {0:.2f}'.format(3*degrees(self.view.alpha[3])))
        self.lab5.setText('beta2  {0:.2f}'.format(degrees(self.view.beta[3])))
        self.lab6.setText('length2 {0:.3} m'.format(3*self.view.lm[3]))
        self.lab7.setText('alpha3 {0:.2f}'.format(3*degrees(self.view.alpha[6])))
        self.lab8.setText('beta3  {0:.2f}'.format(degrees(self.view.beta[6])))
        self.lab9.setText('length3 {0:.3} m'.format(3*self.view.lm[6]))

        self.lab10.setText('x {0:.2f} '.format(self.view.pts[8][18][0]/10))
        self.lab11.setText('y {0:.2f} '.format(-self.view.pts[8][18][1]/10))
        self.lab12.setText('z {0:.2f} '.format(-self.view.pts[8][18][2]/10))

        '''self.lab13.setText('alahpa {0:.1f} '.format(degrees(atan2(self.view.angle[8][1], self.view.angle[8][0]))))
        self.lab14.setText('beta {0:.1f} '.format(degrees(atan2(self.view.angle[8][0], self.view.angle[8][1]))))
        self.lab15.setText('gamma {0:.1f} '.format(degrees(atan2(self.view.angle[8][2], sqrt(self.view.angle[8][0]**2+self.view.angle[8][1]**2)))))'''

        self.lab20.setText('seg1.P1  {0:3.1f} kPa  seg2.P1.{1:3.1f} kPa  seg3.P1  {2:3.1f} kPa'.format(self.view.soft[0].pressureD[0]/1000, self.view.soft[3].pressureD[0]/1000, self.view.soft[6].pressureD[0]/1000))
        self.lab21.setText('seg1.P2  {0:3.1f} kPa  seg2.P2.{1:3.1f} kPa  seg3.P2  {2:3.1f} kPa'.format(self.view.soft[0].pressureD[1]/1000, self.view.soft[3].pressureD[1]/1000, self.view.soft[6].pressureD[1]/1000))
        self.lab22.setText('seg1.P3  {0:3.1f} kPa  seg2.P3.{1:3.1f} kPa  seg3.P3  {2:3.1f} kPa'.format(self.view.soft[0].pressureD[2]/1000, self.view.soft[3].pressureD[2]/1000, self.view.soft[6].pressureD[2]/1000))
        self.lab23.setText('seg1.P4  {0:3.1f} kPa  seg2.P4.{1:3.1f} kPa  seg3.P4  {2:3.1f} kPa'.format(self.view.soft[0].pressureD[3]/1000, self.view.soft[3].pressureD[3]/1000, self.view.soft[6].pressureD[3]/1000))
        self.lab24.setText('seg1.P5  {0:3.1f} kPa  seg2.P5.{1:3.1f} kPa  seg3.P5  {2:3.1f} kPa'.format(self.view.soft[0].pressureD[4]/1000, self.view.soft[3].pressureD[4]/1000, self.view.soft[6].pressureD[4]/1000))
        self.lab25.setText('seg1.P6  {0:3.1f} kPa  seg2.P6.{1:3.1f} kPa  seg3.P6  {2:3.1f} kPa'.format(self.view.soft[0].pressureD[5]/1000, self.view.soft[3].pressureD[5]/1000, self.view.soft[6].pressureD[5]/1000))

    def Combox(self, index):
        print(index)

if __name__ == '__main__':
    x = [0.7854173188675502, 0.8950722011181647, 0.7102987953271219, 0.6672106795297021, 0.27192914257736583, 3.0672900774729914, 7.103244863929998, 6.23299075554193, 7.854489293477538]
    x = [0.95130287, 0.99206756, 0.66422903, 0.29461994, 1.19270756, 2.82771218, 5.45156515, 5.22752532, 7.80757701]
    x = [pi/9, pi/9, pi/9, 0, 0, 0, 4, 4, 4]
    x[6] = 0.403/3
    x[7] = 0.403/3
    x[8] = 0.403/3
    soft1 = softArm(alphaD=x[0], betaD=x[3], lengthD=x[6])
    soft2 = softArm(alphaD=x[1], betaD=0, lengthD=x[7])
    soft3 = softArm(alphaD=x[2], betaD=0, lengthD=x[8])
    soft4 = softArm(alphaD=x[0], betaD=x[4], lengthD=x[6])
    soft5 = softArm(alphaD=x[1], betaD=0, lengthD=x[7])
    soft6 = softArm(alphaD=x[2], betaD=0, lengthD=x[8])
    soft7 = softArm(alphaD=x[0], betaD=x[5], lengthD=x[6])
    soft8 = softArm(alphaD=x[1], betaD=0, lengthD=x[7])
    soft9 = softArm(alphaD=x[2], betaD=0, lengthD=x[8])

    softArms = SoftObject(soft1, soft2, soft3, soft4, soft5, soft6, soft7, soft8, soft9)

    app = QApplication(sys.argv)
    ex = Example(softArms)

    sys.exit(app.exec_())