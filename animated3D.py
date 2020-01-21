# -*- coding: utf-8 -*-
"""
    Animated 3D sinc function
"""
from PyQt5.QtCore import  pyqtSignal, QObject
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QWidget, QApplication
#from PyQt5 import QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
from numpy import cos, sin, pi, degrees,arccos
from math import acos
from myJoyStick import myJoyStick
import time
from scipy import linalg
from softArm import softArm
from SoftObject import SoftObject
from scipy.optimize import least_squares
import sys
import threading

class Signal(QObject):
    ValueSign = pyqtSignal()

class Visualizer(QObject):
    def __init__(self, sf):
        super().__init__()
        #set parameters
        self.joystick = myJoyStick()
        self.joystick.start()
        self.Sign = Signal()
        self.Arms = sf
        self.soft = sf.getArms()
        self.w = gl.GLViewWidget()
        self.w.pan(0,0,-17)
        self.w.opts['distance'] = 60
        self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.flag = 0

        # create the background grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, -10)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, -10)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, 0)
        self.w.addItem(gz)
        self.w.setGeometry(20, 20, 900, 900)

        self.pts = dict()

        self.flag = 0
        self.flag1 = 0
        self.display = 1

        self.dst_pos = [5, 5, 5]
        self.dst_dir = [0, 1, 0]

        self.traces = dict()
        self.incre_alpha = dict()
        self.incre_beta = dict()
        self.incre_length = dict()
        self.pointer = dict()

        self.normal = [.0, .0, .0]
        self.angle = dict()

        self.incre = 0
        self.enlargement = 15

        self.num_seg = sf.num
        self.num_pipe = 3
        self.calculate_flag = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
        self.angle = {0:[0, 1, 0],1:[0, 1, 0],2:[0, 1, 0]}

        for i in range(3):
            for j in range(3):
                self.traces[3*i+j] = gl.GLLinePlotItem(color=pg.glColor((40+30*(i+1), 100+10*(j+1))),
                                                width=2, antialias=True)
                self.w.addItem(self.traces[3*i+j])
        self.circle_show = 1
        if self.circle_show:
            self.create_circle()
        self.alpha = dict()
        self.beta = dict()
        self.lm = dict()

        self.nx = {0:[1, 0, 0]}
        self.nz = {0:[0, 0, 1]}

        self.update()

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QApplication.instance().exec_()
            self.alpha = self.alpha

    def update(self):
        #now = time.time()
        self.move()

        for i in range(self.num_seg):
            if self.num_seg == 2:
                if not i:
                    self.soft[i].SetV(Vx = self.joystick.X, Vy = self.joystick.Y, Vz = self.joystick.Z)
                    self.soft[i].SetW(alphaW = self.joystick.Y*pi/10, betaW = self.joystick.X*pi/10, length = self.joystick.Z)
                elif i == 1:
                    self.soft[i].SetV(Vx=self.joystick.RX, Vy=self.joystick.RY, Vz=self.joystick.TZ)
                    self.soft[i].SetW(alphaW=self.joystick.RY*pi/10, betaW=self.joystick.RX*pi/10,
                                      length=self.joystick.TZ)
            self.alpha[i], self.beta[i], self.lm[i] = self.soft[i].GetABL(self.joystick.flag)
            self.create_Line(i, self.alpha[i], self.beta[i], self.lm[i]*10)

        #self.transfer()
        #print(time.time()-now)
        self.transfer_line()

        if self.incre:
            self.Sign.ValueSign.emit()
            self.incre -= 1

    def create_Line(self, seg, alpha, beta, lm): # 扩展接口：
        theta = np.linspace(0, alpha, 19)
        x = np.array([.0]*19)
        z = np.array([.0]*19)
        for i in range(19):
            # 根据alpha lm计算y=0 平面上图形
            x[i] = lm/alpha*(1-cos(theta[i]))
            z[i] = lm/alpha*sin(theta[i])
        y = np.array([0]*19)
        transB = np.array([[cos(beta), sin(beta), 0],
                  [-sin(beta), cos(beta), 0],
                  [0, 0, 1]])
        self.pts[seg] = np.vstack([x, y, z]).transpose().dot(transB)
        self.nx[seg+1] = [cos(beta)*cos(alpha), sin(beta)*cos(alpha), -sin(alpha)]
        self.nz[seg+1] = [cos(beta)*sin(alpha), sin(beta)*sin(alpha), cos(alpha)]

    def create_circle(self):
        def create(L, r):
            num_res = 100
            pos3 = np.zeros((num_res, num_res, 3))
            pos3[:, :, :2] = np.mgrid[:num_res, :num_res].transpose(1, 2, 0) * [-0.1, 0.1]
            pos3 = pos3.reshape(num_res**2, 3)
            d3 = (pos3 ** 2).sum(axis=1) ** 0.5
            area = L #产生备选点的区域
            ring_res = 0.08 #环粗细
            for i in range(num_res):
                pos3[i * num_res:num_res * (i + 1), 0] = -area + 2*area*i/num_res
                pos3[i * num_res:num_res * (i + 1), 1] = np.linspace(-area, area, num_res)
                pos3[i * num_res:num_res * (i + 1), 2] = 0
            count = 0
            list1 = list()
            rad_ring = r #环圆心距离
            ring = 0.029*10 #环半径
            for i in range(num_res**2):
                if  (ring - ring_res) ** 2 < ((pos3[i, 0]) ** 2 + (pos3[i, 1]-rad_ring) ** 2 )< ring**2  or\
                    (ring - ring_res) ** 2 < ((pos3[i, 0]+rad_ring*0.866) ** 2 + (pos3[i, 1]-rad_ring/2) ** 2)<ring**2 or\
                    (ring - ring_res) ** 2 < ((pos3[i, 0]+rad_ring*0.866) ** 2 + (pos3[i, 1]+rad_ring/2) ** 2)< ring**2  or\
                    (ring - ring_res) ** 2 < ((pos3[i, 0]-rad_ring*0.866) ** 2 + (pos3[i, 1]-rad_ring/2) ** 2)< ring**2  or\
                    (ring - ring_res) ** 2 < ((pos3[i, 0]-rad_ring*0.866) ** 2 + (pos3[i, 1]+rad_ring/2) ** 2)< ring**2  or\
                    (ring - ring_res) ** 2 < ((pos3[i, 0]) ** 2 + (pos3[i, 1]+rad_ring) ** 2)< ring**2  :
                    list1.append(i)
            backup = list()
            for i in list1:
                backup.append(pos3[i])
            return backup

        self.backup = create(L = 2, r = 0.09*10)
        self.backup1 = create(L = 2, r = 0.0615*10)
        self.sp = list()

        color = {0:pg.glColor(80,40,10),1:pg.glColor(80,40,10),2:pg.glColor(80,40,10),3:pg.glColor(80,80,0),4:pg.glColor(80,80,0),5:pg.glColor(80,80,0),6:pg.glColor(0,80,80),7:pg.glColor(0,80,80),8:pg.glColor(0,80,80)}
        for i in range(self.num_seg):
            for j in range(self.num_pipe*i, self.num_pipe*(i+1)):
                if i <= 2:
                    self.sp.append(gl.GLScatterPlotItem(pos=self.backup, size=0.08, pxMode=False, color = color[i]))
                else:
                    self.sp.append(gl.GLScatterPlotItem(pos=self.backup1, size=0.08, pxMode=False, color = color[i]))
                self.w.addItem(self.sp[j])

    def vector_display(self, vector, pos, num):
        if not num in self.pointer.keys():
            self.pointer[num] = gl.GLLinePlotItem(color=pg.glColor((40*num, 50)),
                                                width=2, antialias=True)
            self.w.addItem(self.pointer[num])
        if num%2:
            length = 3
        else:
            length = 6
        x = np.linspace(0, float(vector[0])*length, 10)
        y = np.linspace(0, float(vector[1])*length, 10)
        z = np.linspace(0, float(vector[2])*length, 10)
        pts = np.vstack([x, y, z]).transpose() + pos
        self.pointer[num].setData(pos=pts)

    def update_circle(self, seg):
        #for seg in range(self.num_seg):
        vector1 = np.subtract(self.pts[seg][1], self.pts[seg][0])
        vector2 = np.subtract(self.pts[seg][2], self.pts[seg][0])

        result = -np.cross(vector1, vector2)
        # 化为单位向量
        mod = np.sqrt(np.square(result[0])+np.square(result[1])+np.square(result[2]))
        if mod:
            result = np.divide(result, mod)
        # 旋转轴
        if not seg:
            #母本
            data = self.backup
        elif seg<=2:
            data = np.subtract(self.sp[self.num_pipe*seg-1].pos, self.pts[seg-1][18])
        elif seg == 3:
            data = np.dot(self.backup1, self.spin)
        elif seg > 3:
            data = np.subtract(self.sp[self.num_pipe * seg - 1].pos, self.pts[seg - 1][18])
        for i in range(self.num_pipe):
            spin = -np.array(linalg.expm(np.multiply((i+1)*self.alpha[seg]/self.num_pipe, self.hat(result))))
            if seg==2 and i==2:
                self.spin = -np.array(linalg.expm(np.multiply((i+1)*self.alpha[seg], self.hat(result))))
            self.sp[i+self.num_pipe*seg].setData(pos=np.add(np.dot(data, spin), self.pts[seg][6*(i+1)][:]))

    def hat(self, vector):
        hat = np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
        return hat

    def transfer(self) :
        # 每个seg 两次旋转
        for seg in range(1, self.num_seg):
            #print(seg)
            angle_x = acos(np.dot(self.nx[seg], self.nx[0]))
            #前一个节点的x轴和后一个节点的x轴叉乘
            axis_x = np.cross(self.nx[seg], self.nx[0])
            mod = np.sqrt(axis_x[0]**2+axis_x[1]**2+axis_x[2]**2)
            if mod:
                axis_x = np.divide(axis_x, mod)
            spin_x = np.array(linalg.expm(np.multiply(angle_x, self.hat(axis_x))))

            nz = np.dot(self.nz[0], spin_x)

            angle_z = arccos(np.clip(np.dot(nz, self.nz[seg]), -1.0, 1.0))
            #对比旋转后结果 不符合即反转
            right = 1
            while right:
                spin_z = np.array(linalg.expm(np.multiply(angle_z, self.hat(self.nx[seg]))))
                check = np.dot(nz, spin_z) - self.nz[seg]
                if -0.005<check[0] <0.005 and -0.005<check[1] <0.005 and -0.005<check[2] <0.005:
                    right = 0
                else:
                    angle_z = -angle_z

            self.pts[seg] = np.dot(np.dot(self.pts[seg], spin_x), spin_z)
            self.pts[seg] += self.pts[seg-1][18]

            self.nx[seg+1] = np.dot(np.dot(self.nx[seg+1], spin_x), spin_z)
            self.nz[seg+1] = np.dot(np.dot(self.nz[seg+1], spin_x), spin_z)

            self.angle[seg] = self.nz[seg+1]

        for i in range(self.num_seg):
            for j in range(19):  # 翻转z坐标y坐标
                self.pts[i][j][1] = -self.pts[i][j][1]
                self.pts[i][j][2] = -self.pts[i][j][2]
            self.traces[i].setData(pos=self.pts[i])
        if self.circle_show:
            self.update_circle()
    # 基于向量

    def transfer_line(self):
        def transA(alpha):
            result = np.array(
                [[cos(alpha), 0, -sin(alpha)],
                [0, 1, 0],
                [sin(alpha), 0, cos(alpha)]]
                            )
            return result
        def transB(beta):
            result = np.array(
                [[cos(beta), sin(beta), 0],
                 [-sin(beta), cos(beta), 0],
                 [0, 0, 1]]
            )
            return result
        def transA_(alpha, base):
            result = np.array(
                [[cos(alpha), 0, -sin(alpha), 0],
                [0, 1, 0, 0],
                [sin(alpha), 0, cos(alpha), 0],
                [base[0], base[1], base[2], 1]]
                            )
            return result
        def transB_(beta):
            result = np.array(
                [[cos(beta), sin(beta), 0, 0],
                 [-sin(beta), cos(beta), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            )
            return result
        trans_A = dict()
        trans_B = dict()
        base_ = dict()
        transform = np.eye(4)
        '''now = time.time()
        for seg in range(self.num_seg):
            trans_A[seg] = transA(self.alpha[seg])
            trans_B[seg] = transB(self.beta[seg])
            base_[seg] = [self.lm[seg] /self.alpha[seg] * (1 - cos(self.alpha[seg])),
                    0,
                    self.lm[seg] /self.alpha[seg] * sin(self.alpha[seg])
                        ]

        for seg in range(self.num_seg):
            for i in reversed(range(seg)):
                self.pts[seg] = np.dot(self.pts[seg], trans_A[i])
                self.pts[seg] += base_[i]
                self.pts[seg] = np.dot(self.pts[seg], trans_B[i])

                self.nz[seg + 1] = np.dot(self.nz[seg + 1], trans_A[i])
                self.nz[seg+1] = np.dot(self.nz[seg+1], trans_B[i])
                self.angle[i] = self.nz[i+1]'''

        for seg in range(1, self.num_seg):
            a = np.insert(self.pts[seg], 3, np.array([1.0]*19), axis=1)
            base= [self.lm[seg-1] / self.alpha[seg-1] * (1 - cos(self.alpha[seg-1]))*10,
                          0,
                          self.lm[seg-1] / self.alpha[seg-1] * sin(self.alpha[seg-1])*10
                          ]
            transform = transA_(self.alpha[seg-1], base).dot(transB_(self.beta[seg-1])).dot(transform)
            self.pts[seg] = np.delete(a.dot(transform), 3, axis=1)

        for seg in range(self.num_seg):
            for j in range(19):  # 翻转z坐标y坐标
                self.pts[seg][j][1] = -self.pts[seg][j][1]
                self.pts[seg][j][2] = -self.pts[seg][j][2]

            self.update_circle(seg)
            self.traces[seg].setData(pos=self.pts[seg])

    # 基于方程迭代

    def move(self):
        if self.incre:
            for i in range(3):
                for j in range(3):
                    self.soft[3*i+j].alphaD += self.incre_alpha[i]
                    self.soft[3*i+j].lengthD += self.incre_length[i]
            for i in range(3):
                self.soft[3*i].betaD += self.incre_beta[i]

    def backward_position(self, x, y, z, alpha, beta, gamma):
        #a1 a2 b1 b2 l1 l2
        self.dst_pos = [x, y, z]
        self.dst_dir = [alpha, beta, gamma]
        #self.dst_dir = [cos(rx), cos(ry), cos(rz)]/(cos(rx)**2 + cos(ry)**2 + cos(rz)**2)

        pos_now = [self.alpha[0], self.alpha[3], self.alpha[6],
                   self.beta[0], self.beta[3], self.beta[6],
                   self.lm[0]/self.alpha[0],
                   self.lm[3]/self.alpha[3],
                   self.lm[6]/self.alpha[6]]
        #print(pos_now)
        #h = leastsq(self.position_3, pos_now)

        def test_square(dst, angle, x0): #a1 a2 a3 b1 b2 b3 r1 r2 r3
            def test_3_rad(x):
                a1 = float(x[0])
                a2 = float(x[1])
                a3 = float(x[2])
                b1 = float(x[3])
                b2 = float(x[4])
                b3 = float(x[5])
                rad1 = float(x[6])
                rad2 = float(x[7])
                rad3 = float(x[8])
                result = [-(rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) + (
                            rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(
                        a3)) * sin(b2)) * sin(b1) + (rad1 * (1 - cos(a1)) + (
                            -rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                                rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(
                            a3)) * cos(b2)) * cos(a1) + (rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(
                    b3) + rad3 * sin(a3) * cos(a2)) * sin(a1)) * cos(b1),
                                   (rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) + (
                                               rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(
                                           b3) + rad3 * sin(a2) * sin(a3)) * sin(b2)) * cos(b1) + (
                                               rad1 * (1 - cos(a1)) + (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                                                   rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(
                                               b3) + rad3 * sin(a2) * sin(a3)) * cos(b2)) * cos(a1) + (
                                                           rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(
                                                       b3) + rad3 * sin(a3) * cos(a2)) * sin(a1)) * sin(b1),
                                   rad1 * sin(a1) - (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                                               rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(
                                           b3) + rad3 * sin(a2) * sin(a3)) * cos(b2)) * sin(a1) + (
                                               rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(
                                           a3) * cos(a2)) * cos(a1),
                                   (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(
                                       b2) * sin(b3)) * cos(a1) + (
                                                -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * cos(
                                       b1) - ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(
                                       a3) * sin(b3) * cos(b2)) * sin(b1),
                                   (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(
                                       b2) * sin(b3)) * cos(a1) + (
                                                -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * sin(
                                       b1) + ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(
                                       a3) * sin(b3) * cos(b2)) * cos(b1),
                                   -((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(
                                       b2) * sin(b3)) * sin(a1) + (
                                               -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1) ]
                return result
            def test_3(x):
                a1 = float(x[0])
                a2 = float(x[1])
                a3 = float(x[2])
                b1 = float(x[3])
                b2 = float(x[4])
                b3 = float(x[5])
                rad1 = float(x[6])
                rad2 = float(x[7])
                rad3 = float(x[8])
                result = np.array([-(rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) + (
                            rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(
                        a3)) * sin(b2)) * sin(b1) + (rad1 * (1 - cos(a1)) + (
                            -rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                                rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(
                            a3)) * cos(b2)) * cos(a1) + (rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(
                    b3) + rad3 * sin(a3) * cos(a2)) * sin(a1)) * cos(b1) - dst[0],
                                   (rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) + (
                                               rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(
                                           b3) + rad3 * sin(a2) * sin(a3)) * sin(b2)) * cos(b1) + (
                                               rad1 * (1 - cos(a1)) + (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                                                   rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(
                                               b3) + rad3 * sin(a2) * sin(a3)) * cos(b2)) * cos(a1) + (
                                                           rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(
                                                       b3) + rad3 * sin(a3) * cos(a2)) * sin(a1)) * sin(b1) - dst[1],
                                   rad1 * sin(a1) - (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                                               rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(
                                           b3) + rad3 * sin(a2) * sin(a3)) * cos(b2)) * sin(a1) + (
                                               rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(
                                           a3) * cos(a2)) * cos(a1) - dst[2],
                                   (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(
                                       b2) * sin(b3)) * cos(a1) + (
                                                -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * cos(
                                       b1) - ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(
                                       a3) * sin(b3) * cos(b2)) * sin(b1) - angle[0],
                                   (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(
                                       b2) * sin(b3)) * cos(a1) + (
                                                -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * sin(
                                       b1) + ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(
                                       a3) * sin(b3) * cos(b2)) * cos(b1) - angle[1],
                                   -((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(
                                       b2) * sin(b3)) * sin(a1) + (
                                               -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1) - angle[2],
                                   (x[6] * x[0] - (x[6] * x[0] + x[7] * x[1] + x[8] * x[2]) / 3) ** 2,
                                   (x[7] * x[1] - (x[6] * x[0] + x[7] * x[1] + x[8] * x[2]) / 3) ** 2,
                                   (x[8] * x[2] - (x[6] * x[0] + x[7] * x[1] + x[8] * x[2]) / 3) ** 2])
                return result.astype('float64')

            x0_rosenbrock = np.array(x0).astype('float64')
            res = least_squares(test_3, x0_rosenbrock,
                                bounds=([0, 0, 0, -pi, -pi, -pi, 0.0, 0.0, 0.0], [pi, pi, pi, pi, pi, pi, np.inf, np.inf, np.inf]))

            #print('x:', res.x)
            #print('cost:',res.cost)
            result = [1,1,1,1,1,1,1,1,1]
            for i in range(9):
                result[i] = res.x[i]

            result[6] = result[6] * result[0]
            result[7] = result[7] * result[1]
            result[8] = result[8] * result[2]
            '''print('result:', result)
            print('position:',test_3_rad(res.x))'''
            return result

        result = test_square(self.dst_pos, self.dst_dir, pos_now)

        self.incre = 20

        # 3X3 控制模式
        for i in range(3):
            self.incre_alpha[i] = (result[i] - self.alpha[3*i]*3)/self.incre/3
            self.incre_beta[i] = (result[3+i] - self.beta[3*i])/self.incre
            self.incre_length[i] = (result[6+i] - self.lm[3*i]*3)/self.incre/3

        return 0

    def animation(self):
        if self.flag == 0:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(30)
            #self.start()
            self.flag = 1
        else:
            self.timer.stop()
            self.flag = 0

class parameter:
    def __init__(self):
        self.alpha = 0
        self.beta = 0
        self.length = 0

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    x = [0.7854173188675502, 0.8950722011181647, 0.7102987953271219, 0.6672106795297021, 0.27192914257736583,
         3.0672900774729914, 7.103244863929998, 6.23299075554193, 7.854489293477538]
    x = [0.95130287, 0.99206756, 0.66422903, 0.29461994, 1.19270756, 2.82771218, 5.45156515, 5.22752532, 7.80757701]
    x = [pi / 9, pi / 9, pi / 9, 0, 0, 0, 4, 4, 4]
    '''x[6] = x[6] * x[0]
    x[7] = x[7] * x[1]
    x[8] = x[8] * x[2]'''
    soft1 = softArm(alphaD=x[0], betaD=x[3], lengthD=x[6])
    soft2 = softArm(alphaD=x[1], betaD=x[4], lengthD=x[7])
    soft3 = softArm(alphaD=x[2], betaD=x[5], lengthD=x[8])
    soft4 = softArm(alphaD=x[0], betaD=x[3], lengthD=x[6])
    soft5 = softArm(alphaD=x[1], betaD=x[4], lengthD=x[7])
    soft6 = softArm(alphaD=x[2], betaD=x[5], lengthD=x[8])
    soft7 = softArm(alphaD=x[0], betaD=x[3], lengthD=x[6])
    soft8 = softArm(alphaD=x[1], betaD=x[4], lengthD=x[7])
    soft9 = softArm(alphaD=x[2], betaD=x[5], lengthD=x[8])

    softArms = SoftObject(soft1, soft2, soft3, soft4, soft5, soft6, soft7, soft8, soft9)
    app = QApplication(sys.argv)
    test =  Visualizer(softArms)
    test.w.show()
    #test.animation()
    #test.backward_position(8, -8, 8, 0, 0, -1)

    sys.exit(app.exec_())


