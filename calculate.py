#from sympy import pi,cos,sin,symbols,solve
#from math import sin, cos
from sympy import symbols, sin, cos, diff, tan, Matrix
#from numpy import sin,cos
import time
import numpy as np
from math import pi
from scipy.optimize import fsolve, leastsq, least_squares, root
#from numpy import sin, cos, pi
from scipy.optimize import minimize

def position_3(x):
    a1 = float(x[0])
    a2 = float(x[1])
    a3 = float(x[2])
    b1 = float(x[3])
    b2 = float(x[4])
    b3 = float(x[5])
    lm1 = float(x[6])
    lm2 = float(x[6])
    lm3 = float(x[6])
    return [-((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*sin(b2) + lm3*(1 - cos(a3))*sin(b3)*cos(b2)/a3)*sin(b1) +
            (((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*cos(a1) +
             (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 + lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*sin(a1) + lm1*(1 - cos(a1))/a1)*cos(b1)-5,
 ((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*sin(b2) + lm3*(1 - cos(a3))*sin(b3)*cos(b2)/a3)*cos(b1) + (((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*cos(a1) +
  (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 + lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*sin(a1) + lm1*(1 - cos(a1))/a1)*sin(b1),
 -((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*sin(a1) + (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 +
            lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*cos(a1) + lm1*sin(a1)/a1,
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*cos(b1) -
            ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*sin(b1)-1,
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*sin(b1) +
            ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*cos(b1),
              -((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*sin(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*cos(a1),
            0
            ]

def position_3_origin(x):
    a1 = float(x[0])
    a2 = float(x[1])
    a3 = float(x[2])
    b1 = float(x[3])
    b2 = float(x[4])
    b3 = float(x[5])
    lm1 = float(x[6])
    lm2 = float(x[6])
    lm3 = float(x[6])
    return [-((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*sin(b2) + lm3*(1 - cos(a3))*sin(b3)*cos(b2)/a3)*sin(b1) +
            (((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*cos(a1) + (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 + lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*sin(a1) + lm1*(1 - cos(a1))/a1)*cos(b1),
 ((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*sin(b2) + lm3*(1 - cos(a3))*sin(b3)*cos(b2)/a3)*cos(b1) + (((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*cos(a1) + (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 + lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*sin(a1) + lm1*(1 - cos(a1))/a1)*sin(b1),
 -((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*sin(a1) + (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 + lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*cos(a1) + lm1*sin(a1)/a1,
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*cos(b1) - ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*sin(b1),
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*sin(b1) + ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*cos(b1),
              -((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*sin(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*cos(a1),
            0
            ]

def derivation(seg, switch):
    #推导seg=num 情况下的末端状态方程
    a = dict()
    b = dict()
    rad = dict()
    lm = dict()
    for i in range(seg):
        sym_a = 'a{0}'.format(i+1)
        sym_b = 'b{0}'.format(i+1)
        sym_rad = 'rad{0}'.format(i+1)
        sym_lm = 'lm{0}'.format(i+1)
        a[i] = symbols(sym_a)
        b[i] = symbols(sym_b)
        rad[i] = symbols(sym_rad)
        lm[i] = symbols(sym_lm)

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
    if switch:
        pts = np.dot([
                rad[seg-1] * (1 - cos(a[seg-1])),
                0,
                rad[seg-1] * sin(a[seg-1])
            ], transB(b[seg-1]))
        nz = [cos(b[seg-1]) * sin(a[seg-1]), sin(b[seg-1]) * sin(a[seg-1]), cos(a[seg-1])]
        if seg-1:
            for i in reversed(range(seg-1)):
                pts = np.dot(pts, transA(a[i]))
                nz = np.dot(nz, transA(a[i]))
                pts += [
                    rad[i] * (1 - cos(a[i])),
                    0,
                    rad[i] * sin(a[i])
                ]
                pts = np.dot(pts, transB(b[i]))
                nz = np.dot(nz, transB(b[i]))
    if not switch:
        pts = np.dot([
            lm[seg-1]/a[seg-1] * (1 - cos(a[seg - 1])),
            0,
            lm[seg-1]/a[seg-1] * sin(a[seg - 1])
        ], transB(b[seg - 1]))
        nz = [cos(b[seg - 1]) * sin(a[seg - 1]), sin(b[seg - 1]) * sin(a[seg - 1]), cos(a[seg - 1])]
        if seg - 1:
            for i in reversed(range(seg - 1)):
                pts = np.dot(pts, transA(a[i]))
                nz = np.dot(nz, transA(a[i]))
                pts += [
                    lm[i]/a[i] * (1 - cos(a[i])),
                    0,
                    lm[i]/a[i] * sin(a[i])
                ]
                pts = np.dot(pts, transB(b[i]))
                nz = np.dot(nz, transB(b[i]))
    arg = Matrix([ a, b, rad])
    print('jacobian xyz:', Matrix(pts).jacobian(arg))
    print('jacobian nz:', Matrix(nz).jacobian(arg))

    print("pts:",pts)
    print("nz:",nz)

def test_3_origin(x):
    a1 = float(x[0])
    a2 = float(x[1])
    a3 = float(x[2])
    b1 = float(x[3])
    b2 = float(x[4])
    b3 = float(x[5])
    lm1 = float(x[6])
    lm2 = float(x[6])
    lm3 = float(x[6])
    result = [-((lm2*(1 - cos(a2))/a2 + lm2*(1 - cos(a3))*cos(a2)*cos(b3)/a2 + lm2*sin(a2)*sin(a3)/a2)*sin(b2) + lm2*(1 - cos(a3))*sin(b3)*cos(b2)/a2)*sin(b1) + (((lm2*(1 - cos(a2))/a2 + lm2*(1 - cos(a3))*cos(a2)*cos(b3)/a2 + lm2*sin(a2)*sin(a3)/a2)*cos(b2) - lm2*(1 - cos(a3))*sin(b2)*sin(b3)/a2)*cos(a1) + (-lm2*(1 - cos(a3))*sin(a2)*cos(b3)/a2 + lm2*sin(a2)/a2 + lm2*sin(a3)*cos(a2)/a2)*sin(a1) + lm1*(1 - cos(a1))/a1)*cos(b1),
 ((lm2*(1 - cos(a2))/a2 + lm2*(1 - cos(a3))*cos(a2)*cos(b3)/a2 + lm2*sin(a2)*sin(a3)/a2)*sin(b2) + lm2*(1 - cos(a3))*sin(b3)*cos(b2)/a2)*cos(b1) + (((lm2*(1 - cos(a2))/a2 + lm2*(1 - cos(a3))*cos(a2)*cos(b3)/a2 + lm2*sin(a2)*sin(a3)/a2)*cos(b2) - lm2*(1 - cos(a3))*sin(b2)*sin(b3)/a2)*cos(a1) + (-lm2*(1 - cos(a3))*sin(a2)*cos(b3)/a2 + lm2*sin(a2)/a2 + lm2*sin(a3)*cos(a2)/a2)*sin(a1) + lm1*(1 - cos(a1))/a1)*sin(b1),
 -((lm2*(1 - cos(a2))/a2 + lm2*(1 - cos(a3))*cos(a2)*cos(b3)/a2 + lm2*sin(a2)*sin(a3)/a2)*cos(b2) - lm2*(1 - cos(a3))*sin(b2)*sin(b3)/a2)*sin(a1) + (-lm2*(1 - cos(a3))*sin(a2)*cos(b3)/a2 + lm2*sin(a2)/a2 + lm2*sin(a3)*cos(a2)/a2)*cos(a1) + lm1*sin(a1)/a1,
              (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * cos(a1) + (
                          -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * cos(b1) - (
                          (sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(a3) * sin(b3) * cos(
                      b2)) * sin(b1),
              (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * cos(a1) + (
                          -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * sin(b1) + (
                          (sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(a3) * sin(b3) * cos(
                      b2)) * cos(b1),
              - ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * sin(
                  a1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1)
            ,0]
    return result

def test_3_(x):
    a1 = float(x[0])
    a2 = float(x[1])
    a3 = float(x[2])
    b1 = float(x[3])
    b2 = float(x[4])
    b3 = float(x[5])
    lm1 = float(x[6])
    lm2 = float(x[6])
    lm3 = float(x[6])
    result = [-((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*sin(b2) + lm3*(1 - cos(a3))*sin(b3)*cos(b2)/a3)*sin(b1) +
            (((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*cos(a1) + (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 + lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*sin(a1) + lm1*(1 - cos(a1))/a1)*cos(b1) - 5,
 ((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*sin(b2) + lm3*(1 - cos(a3))*sin(b3)*cos(b2)/a3)*cos(b1) + (((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*cos(a1) + (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 + lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*sin(a1) + lm1*(1 - cos(a1))/a1)*sin(b1) -5,
 -((lm3*(1 - cos(a3))*cos(a2)*cos(b3)/a3 + lm3*sin(a2)*sin(a3)/a3 + lm2*(1 - cos(a2))/a2)*cos(b2) - lm3*(1 - cos(a3))*sin(b2)*sin(b3)/a3)*sin(a1) + (-lm3*(1 - cos(a3))*sin(a2)*cos(b3)/a3 + lm3*sin(a3)*cos(a2)/a3 + lm2*sin(a2)/a2)*cos(a1) + lm1*sin(a1)/a1 - 5,
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*cos(b1) - ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*sin(b1) - 1,
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*sin(b1) + ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*cos(b1),
              -((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*sin(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*cos(a1)
            ,0]
    return result

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
    result = [-(rad3*(1 - cos(a3))*sin(b3)*cos(b2) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*sin(b2))*sin(b1) + (rad1*(1 - cos(a1)) + (-rad3*(1 - cos(a3))*sin(b2)*sin(b3) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*cos(b2))*cos(a1) + (rad2*sin(a2) - rad3*(1 - cos(a3))*sin(a2)*cos(b3) + rad3*sin(a3)*cos(a2))*sin(a1))*cos(b1),
 (rad3*(1 - cos(a3))*sin(b3)*cos(b2) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*sin(b2))*cos(b1) + (rad1*(1 - cos(a1)) + (-rad3*(1 - cos(a3))*sin(b2)*sin(b3) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*cos(b2))*cos(a1) + (rad2*sin(a2) - rad3*(1 - cos(a3))*sin(a2)*cos(b3) + rad3*sin(a3)*cos(a2))*sin(a1))*sin(b1),
 rad1*sin(a1) - (-rad3*(1 - cos(a3))*sin(b2)*sin(b3) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*cos(b2))*sin(a1) + (rad2*sin(a2) - rad3*(1 - cos(a3))*sin(a2)*cos(b3) + rad3*sin(a3)*cos(a2))*cos(a1),
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*cos(b1) - ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*sin(b1),
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*sin(b1) + ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*cos(b1),
              -((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*sin(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*cos(a1)
            ,0]
    return result

def test_2(x):
    a1 = float(x[0])
    a2 = float(x[1])
    b1 = float(x[2])
    b2 = float(x[3])
    lm1 = float(x[4])
    lm2 = float(x[5])
    return [(lm2*(1 - cos(a2))*cos(a1)*cos(b2)/a2 + lm2*sin(a1)*sin(a2)/a2 + lm1*(1-cos(a1))/a1)*cos(b1)-lm2*(1-cos(a2))*sin(b1)*sin(b2)/a2 -5,
 (lm2*(1-cos(a2))*cos(a1)*cos(b2)/a2 + lm2*sin(a1)*sin(a2)/a2 + lm1*(1-cos(a1))/a1)*sin(b1) + lm2*(1 - cos(a2))*sin(b2)*cos(b1)/a2 - 5,
 -lm2*(1-cos(a2))*sin(a1)*cos(b2)/a2 + lm2*sin(a2)*cos(a1)/a2 + lm1*sin(a1)/a1 - 5,
 (sin(a1)*cos(a2) + sin(a2)*cos(a1)*cos(b2))*cos(b1) - sin(a2)*sin(b1)*sin(b2) - 1,
 (sin(a1)*cos(a2) + sin(a2)*cos(a1)*cos(b2))*sin(b1) + sin(a2)*sin(b2)*cos(b1),
 -sin(a1)*sin(a2)*cos(b2) + cos(a1)*cos(a2)]

def test_2_origin(x):
    a1 = float(x[0])
    a2 = float(x[1])
    b1 = float(x[2])
    b2 = float(x[3])
    lm1 = float(x[4])
    lm2 = float(x[5])
    return [(lm2*(1 - cos(a2))*cos(a1)*cos(b2)/a2 + lm2*sin(a1)*sin(a2)/a2 + lm1*(1-cos(a1))/a1)*cos(b1)-lm2*(1-cos(a2))*sin(b1)*sin(b2)/a2,
 (lm2*(1-cos(a2))*cos(a1)*cos(b2)/a2 + lm2*sin(a1)*sin(a2)/a2 + lm1*(1-cos(a1))/a1)*sin(b1) + lm2*(1 - cos(a2))*sin(b2)*cos(b1)/a2,
 -lm2*(1-cos(a2))*sin(a1)*cos(b2)/a2 + lm2*sin(a2)*cos(a1)/a2 + lm1*sin(a1)/a1,
 (sin(a1)*cos(a2) + sin(a2)*cos(a1)*cos(b2))*cos(b1) - sin(a2)*sin(b1)*sin(b2) ,
 (sin(a1)*cos(a2) + sin(a2)*cos(a1)*cos(b2))*sin(b1) + sin(a2)*sin(b2)*cos(b1),
 -sin(a1)*sin(a2)*cos(b2) + cos(a1)*cos(a2)]

def test_1(x):
    return [lm1 * (1 - cos(a1)) * cos(b1) / a1, # Xb
            lm1 * (1 - cos(a1)) * sin(b1) / a1, # Yb
            lm1 * sin(a1) / a1, # Zb
            sin(a1) * cos(b1), # nx
            sin(a1) * sin(b1), # ny
            cos(a1)] # nz

def rank(a1=pi/2, a2=pi/2, b1=pi/2, b2=pi/2, l1=4, l2=4):
    f11 = (-l2 * (1 - cos(a2)) ** 2 * sin(a1) * cos(b2) / a2 ** 2 + l2 * (1 - cos(a2)) * sin(a2) * cos(
        a1) / a2 ** 2 + 2 * l1 * (1 - cos(a1)) * sin(a1) / a1 ** 2 - 2 * l1 * (1 - cos(a1)) ** 2 / a1 ** 3) * cos(b1)
    f12 = -(l2 * (1 - cos(a2)) ** 2 * cos(a1) * cos(b2) / a2 ** 2 + l2 * (1 - cos(a2)) * sin(a1) * sin(
        a2) / a2 ** 2 + l1 * (1 - cos(a1)) ** 2 / a1 ** 2) * sin(b1) - l2 * (1 - cos(a2)) ** 2 * sin(b2) * cos(
        b1) / a2 ** 2
    f13 = (l2 * (1 - cos(a2)) * sin(a1) * cos(a2) / a2 ** 2 + 2 * l2 * (1 - cos(a2)) * sin(a2) * cos(a1) * cos(
        b2) / a2 ** 2 + l2 * sin(a1) * sin(a2) ** 2 / a2 ** 2 - 2 * l2 * (1 - cos(a2)) ** 2 * cos(a1) * cos(
        b2) / a2 ** 3 - 2 * l2 * (1 - cos(a2)) * sin(a1) * sin(a2) / a2 ** 3) * cos(b1) - 2 * l2 * (1 - cos(a2)) * sin(
        a2) * sin(b1) * sin(b2) / a2 ** 2 + 2 * l2 * (1 - cos(a2)) ** 2 * sin(b1) * sin(b2) / a2 ** 3
    f14 = (l2 * (1 - cos(a2)) * sin(a1) * cos(a2) / a2 ** 2 + 2 * l2 * (1 - cos(a2)) * sin(a2) * cos(a1) * cos(
        b2) / a2 ** 2 + l2 * sin(a1) * sin(a2) ** 2 / a2 ** 2 - 2 * l2 * (1 - cos(a2)) ** 2 * cos(a1) * cos(
        b2) / a2 ** 3 - 2 * l2 * (1 - cos(a2)) * sin(a1) * sin(a2) / a2 ** 3) * cos(b1) - 2 * l2 * (1 - cos(a2)) * sin(
        a2) * sin(b1) * sin(b2) / a2 ** 2 + 2 * l2 * (1 - cos(a2)) ** 2 * sin(b1) * sin(b2) / a2 ** 3
    f17 = (1 - cos(a1)) ** 2 * cos(b1) / a1 ** 2
    f18 = ((1 - cos(a2)) ** 2 * cos(a1) * cos(b2) / a2 ** 2 + (1 - cos(a2)) * sin(a1) * sin(a2) / a2 ** 2) * cos(b1) - (
                1 - cos(a2)) ** 2 * sin(b1) * sin(b2) / a2 ** 2

    f21 = (-l2 * (1 - cos(a2)) ** 2 * sin(a1) * cos(b2) / a2 ** 2 + l2 * (1 - cos(a2)) * sin(a2) * cos(
        a1) / a2 ** 2 + 2 * l1 * (1 - cos(a1)) * sin(a1) / a1 ** 2 - 2 * l1 * (1 - cos(a1)) ** 2 / a1 ** 3) * sin(b1)
    f22 = (l2 * (1 - cos(a2)) ** 2 * cos(a1) * cos(b2) / a2 ** 2 + l2 * (1 - cos(a2)) * sin(a1) * sin(
        a2) / a2 ** 2 + l1 * (1 - cos(a1)) ** 2 / a1 ** 2) * cos(b1) - l2 * (1 - cos(a2)) ** 2 * sin(b1) * sin(
        b2) / a2 ** 2
    f23 = (l2 * (1 - cos(a2)) * sin(a1) * cos(a2) / a2 ** 2 + 2 * l2 * (1 - cos(a2)) * sin(a2) * cos(a1) * cos(
        b2) / a2 ** 2 + l2 * sin(a1) * sin(a2) ** 2 / a2 ** 2 - 2 * l2 * (1 - cos(a2)) ** 2 * cos(a1) * cos(
        b2) / a2 ** 3 - 2 * l2 * (1 - cos(a2)) * sin(a1) * sin(a2) / a2 ** 3) * sin(b1) + 2 * l2 * (1 - cos(a2)) * sin(
        a2) * sin(b2) * cos(b1) / a2 ** 2 - 2 * l2 * (1 - cos(a2)) ** 2 * sin(b2) * cos(b1) / a2 ** 3
    f24 = -l2 * (1 - cos(a2)) ** 2 * sin(b1) * sin(b2) * cos(a1) / a2 ** 2 + l2 * (1 - cos(a2)) ** 2 * cos(b1) * cos(
        b2) / a2 ** 2
    f27 = (1 - cos(a1)) ** 2 * sin(b1) / a1 ** 2
    f28 = ((1 - cos(a2)) ** 2 * cos(a1) * cos(b2) / a2 ** 2 + (1 - cos(a2)) * sin(a1) * sin(a2) / a2 ** 2) * sin(b1) + (
                1 - cos(a2)) ** 2 * sin(b2) * cos(b1) / a2 ** 2

    f31 = -l2 * (1 - cos(a2)) ** 2 * cos(a1) * cos(b2) / a2 ** 2 - l2 * (1 - cos(a2)) * sin(a1) * sin(
        a2) / a2 ** 2 + l1 * (1 - cos(a1)) * cos(a1) / a1 ** 2 + l1 * sin(a1) ** 2 / a1 ** 2 - 2 * l1 * (
                      1 - cos(a1)) * sin(a1) / a1 ** 3
    f32 = 0
    f33 = -2 * l2 * (1 - cos(a2)) * sin(a1) * sin(a2) * cos(b2) / a2 ** 2 + l2 * (1 - cos(a2)) * cos(a1) * cos(
        a2) / a2 ** 2 + l2 * sin(a2) ** 2 * cos(a1) / a2 ** 2 + 2 * l2 * (1 - cos(a2)) ** 2 * sin(a1) * cos(
        b2) / a2 ** 3 - 2 * l2 * (1 - cos(a2)) * sin(a2) * cos(a1) / a2 ** 3
    f34 = l2 * (1 - cos(a2)) ** 2 * sin(a1) * sin(b2) / a2 ** 2
    f37 = (1 - cos(a1)) * sin(a1) / a1 ** 2
    f38 = -(1 - cos(a2)) ** 2 * sin(a1) * cos(b2) / a2 ** 2 + (1 - cos(a2)) * sin(a2) * cos(a1) / a2 ** 2

    f41 = (-sin(a1) * sin(a2) * cos(b2) + cos(a1) * cos(a2)) * cos(b1)
    f42 = -(sin(a1) * cos(a2) + sin(a2) * cos(a1) * cos(b2)) * sin(b1) - sin(a2) * sin(b2) * cos(b1)
    f43 = (-sin(a1) * sin(a2) + cos(a1) * cos(a2) * cos(b2)) * cos(b1) - sin(b1) * sin(b2) * cos(a2)
    f44 = (-sin(a1) * sin(a2) + cos(a1) * cos(a2) * cos(b2)) * cos(b1) - sin(b1) * sin(b2) * cos(a2)
    f47 = 0
    f48 = 0

    f51 = (-sin(a1) * sin(a2) * cos(b2) + cos(a1) * cos(a2)) * sin(b1)
    f52 = (sin(a1) * cos(a2) + sin(a2) * cos(a1) * cos(b2)) * cos(b1) - sin(a2) * sin(b1) * sin(b2)
    f53 = (-sin(a1) * sin(a2) + cos(a1) * cos(a2) * cos(b2)) * sin(b1) + sin(b2) * cos(a2) * cos(b1)
    f54 = -sin(a2) * sin(b1) * sin(b2) * cos(a1) + sin(a2) * cos(b1) * cos(b2)
    f57 = 0
    f58 = 0

    f61 = -sin(a1) * cos(a2) - sin(a2) * cos(a1) * cos(b2)
    f62 = 0
    f63 = -sin(a1) * cos(a2) * cos(b2) - sin(a2) * cos(a1)
    f64 = sin(a1) * sin(a2) * sin(b2)
    f67 = 0
    f68 = 0

    martix = np.array([
        [f11, f12, f13, f14, f17, f18],
        [f21, f22, f23, f24, f27, f28],
        [f31, f32, f33, f34, f37, f38],
        [f41, f42, f43, f44, f47, f48],
        [f51, f52, f53, f54, f57, f58],
        [f61, f62, f63, f64, f67, f68]
         ])

    rank = np.linalg.matrix_rank(martix)
    print('rank:', rank)

def mini():
    dst = [12,6,8]
    angle = [0,1,0]
    base = [1,0,0]
    x0 = np.array([pi/4, pi/4, pi/4, 0, 0, 0, 4])

    def test_3_(x):
        a1 = float(x[0])
        a2 = float(x[1])
        a3 = float(x[2])
        b1 = float(x[3])
        b2 = float(x[4])
        b3 = float(x[5])
        lm1 = float(x[6])
        lm2 = float(x[6])
        lm3 = float(x[6])
        result = [-((lm3 * (1 - cos(a3)) * cos(a2) * cos(b3) / a3 + lm3 * sin(a2) * sin(a3) / a3 + lm2 * (
                    1 - cos(a2)) / a2) * sin(b2) + lm3 * (1 - cos(a3)) * sin(b3) * cos(b2) / a3) * sin(b1) +
                  (((lm3 * (1 - cos(a3)) * cos(a2) * cos(b3) / a3 + lm3 * sin(a2) * sin(a3) / a3 + lm2 * (
                              1 - cos(a2)) / a2) * cos(b2) - lm3 * (1 - cos(a3)) * sin(b2) * sin(b3) / a3) * cos(a1) + (
                               -lm3 * (1 - cos(a3)) * sin(a2) * cos(b3) / a3 + lm3 * sin(a3) * cos(a2) / a3 + lm2 * sin(
                           a2) / a2) * sin(a1) + lm1 * (1 - cos(a1)) / a1) * cos(b1) - dst[0],
                  ((lm3 * (1 - cos(a3)) * cos(a2) * cos(b3) / a3 + lm3 * sin(a2) * sin(a3) / a3 + lm2 * (
                              1 - cos(a2)) / a2) * sin(b2) + lm3 * (1 - cos(a3)) * sin(b3) * cos(b2) / a3) * cos(b1) + (
                              ((lm3 * (1 - cos(a3)) * cos(a2) * cos(b3) / a3 + lm3 * sin(a2) * sin(a3) / a3 + lm2 * (
                                          1 - cos(a2)) / a2) * cos(b2) - lm3 * (1 - cos(a3)) * sin(b2) * sin(
                                  b3) / a3) * cos(a1) + (
                                          -lm3 * (1 - cos(a3)) * sin(a2) * cos(b3) / a3 + lm3 * sin(a3) * cos(
                                      a2) / a3 + lm2 * sin(a2) / a2) * sin(a1) + lm1 * (1 - cos(a1)) / a1) * sin(
                      b1) - dst[1],
                  -((lm3 * (1 - cos(a3)) * cos(a2) * cos(b3) / a3 + lm3 * sin(a2) * sin(a3) / a3 + lm2 * (
                              1 - cos(a2)) / a2) * cos(b2) - lm3 * (1 - cos(a3)) * sin(b2) * sin(b3) / a3) * sin(a1) + (
                              -lm3 * (1 - cos(a3)) * sin(a2) * cos(b3) / a3 + lm3 * sin(a3) * cos(a2) / a3 + lm2 * sin(
                          a2) / a2) * cos(a1) + lm1 * sin(a1) / a1 - dst[2],
                  (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * cos(
                      a1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * cos(b1) - (
                              (sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(a3) * sin(b3) * cos(
                          b2)) * sin(b1) - angle[0],
                  (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * cos(
                      a1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * sin(b1) + (
                              (sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(a3) * sin(b3) * cos(
                          b2)) * cos(b1) - angle[1],
                  -((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * sin(
                      a1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1) - angle[2]
            , 0]
        return result

    result = leastsq(test_3_, x0)
    print("===================")
    print()
    print("求解函数名称:", leastsq.__name__)
    print("解：", result[0])
    print("各向量值：", test_3(result[0]))

def test_square():
    dst = [8, 8, 8]
    angle = [0.5773502691894317, 0.5773502691894317, 0.5773502691894317]

    def test_3_(x):
        a1 = float(x[0])
        a2 = float(x[1])
        a3 = float(x[2])
        b1 = float(x[3])
        b2 = float(x[4])
        b3 = float(x[5])
        rad1 = float(x[6])
        rad2 = float(x[7])
        rad3 = float(x[8])
        result = np.array([-(rad3*(1 - cos(a3))*sin(b3)*cos(b2) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*sin(b2))*sin(b1) + (rad1*(1 - cos(a1)) + (-rad3*(1 - cos(a3))*sin(b2)*sin(b3) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*cos(b2))*cos(a1) + (rad2*sin(a2) - rad3*(1 - cos(a3))*sin(a2)*cos(b3) + rad3*sin(a3)*cos(a2))*sin(a1))*cos(b1)-dst[0],
 (rad3*(1 - cos(a3))*sin(b3)*cos(b2) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*sin(b2))*cos(b1) + (rad1*(1 - cos(a1)) + (-rad3*(1 - cos(a3))*sin(b2)*sin(b3) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*cos(b2))*cos(a1) + (rad2*sin(a2) - rad3*(1 - cos(a3))*sin(a2)*cos(b3) + rad3*sin(a3)*cos(a2))*sin(a1))*sin(b1)-dst[1],
 rad1*sin(a1) - (-rad3*(1 - cos(a3))*sin(b2)*sin(b3) + (rad2*(1 - cos(a2)) + rad3*(1 - cos(a3))*cos(a2)*cos(b3) + rad3*sin(a2)*sin(a3))*cos(b2))*sin(a1) + (rad2*sin(a2) - rad3*(1 - cos(a3))*sin(a2)*cos(b3) + rad3*sin(a3)*cos(a2))*cos(a1)-dst[2],
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*cos(b1) - ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*sin(b1)-angle[0],
              (((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*cos(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*sin(a1))*sin(b1) + ((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*sin(b2) + sin(a3)*sin(b3)*cos(b2))*cos(b1)-angle[1],
              -((sin(a2)*cos(a3) + sin(a3)*cos(a2)*cos(b3))*cos(b2) - sin(a3)*sin(b2)*sin(b3))*sin(a1) + (-sin(a2)*sin(a3)*cos(b3) + cos(a2)*cos(a3))*cos(a1)-angle[2],
                           (x[6] * x[0] - (x[6] * x[0] + x[7] * x[1] + x[8] * x[2]) / 3) ** 2,
                           (x[7] * x[1] - (x[6] * x[0] + x[7] * x[1] + x[8] * x[2]) / 3) ** 2,
                           (x[8] * x[2] - (x[6] * x[0] + x[7] * x[1] + x[8] * x[2]) / 3) ** 2])
        return result.astype('float64')
    def test_2(x): # x = a1 a2 b1 b2 l1 l2
        return np.array([
            (x[5] * (1 - cos(x[1])) * cos(x[0]) * cos(x[3]) / x[1] + x[5] * sin(x[0]) * sin(x[1]) / x[1] + x[4] * (
                        1 - cos(x[0])) / x[0]) * cos(x[2]) - x[5] * (1 - cos(x[1])) * sin(x[2]) * sin(x[3]) / x[1] - 5,
            (x[5] * (1 - cos(x[1])) * cos(x[0]) * cos(x[3]) / x[1] + x[5] * sin(x[0]) * sin(x[1]) / x[1] + x[4] * (
                        1 - cos(x[0])) / x[0]) * sin(x[2]) + x[5] * (1 - cos(x[1])) * sin(x[3]) * cos(x[2]) / x[1] - 5,
            -x[5] * (1 - cos(x[1])) * sin(x[0]) * cos(x[3]) / x[1] + x[5] * sin(x[1]) * cos(x[0]) / x[1] + x[4] * sin(x[0]) / x[0] - 5,
            (sin(x[0]) * cos(x[1]) + sin(x[1]) * cos(x[0]) * cos(x[3])) * cos(x[2]) - sin(x[1]) * sin(x[2]) * sin(x[3]) - 1,
            (sin(x[0]) * cos(x[1]) + sin(x[1]) * cos(x[0]) * cos(x[3])) * sin(x[2]) + sin(x[1]) * sin(x[3]) * cos(x[2]),
            -sin(x[0]) * sin(x[1]) * cos(x[3]) + cos(x[0]) * cos(x[1])
        ]).astype('float64')
    def jac_rosenbrock(x):
        return np.array([
            [(-x[5] * (1 - cos(x[1])) ** 2 * sin(x[0]) * cos(x[3]) / x[1] ** 2 + x[5] * (1 - cos(x[1])) * sin(x[1]) * cos(
        x[0]) / x[1] ** 2 + 2 * x[4] * (1 - cos(x[0])) * sin(x[0]) / x[0] ** 2 - 2 * x[4] * (1 - cos(x[0])) ** 2 / x[0] ** 3) * cos(x[2]),
    -(x[5] * (1 - cos(x[1])) ** 2 * cos(x[0]) * cos(x[3]) / x[1] ** 2 + x[5] * (1 - cos(x[1])) * sin(x[0]) * sin(
        x[1]) / x[1] ** 2 + x[4] * (1 - cos(x[0])) ** 2 / x[0] ** 2) * sin(x[2]) - x[5] * (1 - cos(x[1])) ** 2 * sin(x[3]) * cos(
        x[2]) / x[1] ** 2,
    (x[5] * (1 - cos(x[1])) * sin(x[0]) * cos(x[1]) / x[1] ** 2 + 2 * x[5] * (1 - cos(x[1])) * sin(x[1]) * cos(x[0]) * cos(
        x[3]) / x[1] ** 2 + x[5] * sin(x[0]) * sin(x[1]) ** 2 / x[1] ** 2 - 2 * x[5] * (1 - cos(x[1])) ** 2 * cos(x[0]) * cos(
        x[3]) / x[1] ** 3 - 2 * x[5] * (1 - cos(x[1])) * sin(x[0]) * sin(x[1]) / x[1] ** 3) * cos(x[2]) - 2 * x[5] * (1 - cos(x[1])) * sin(
        x[1]) * sin(x[2]) * sin(x[3]) / x[1] ** 2 + 2 * x[5] * (1 - cos(x[1])) ** 2 * sin(x[2]) * sin(x[3]) / x[1] ** 3,
    (x[5] * (1 - cos(x[1])) * sin(x[0]) * cos(x[1]) / x[1] ** 2 + 2 * x[5] * (1 - cos(x[1])) * sin(x[1]) * cos(x[0]) * cos(
        x[3]) / x[1] ** 2 + x[5] * sin(x[0]) * sin(x[1]) ** 2 / x[1] ** 2 - 2 * x[5] * (1 - cos(x[1])) ** 2 * cos(x[0]) * cos(
        x[3]) / x[1] ** 3 - 2 * x[5] * (1 - cos(x[1])) * sin(x[0]) * sin(x[1]) / x[1] ** 3) * cos(x[2]) - 2 * x[5] * (1 - cos(x[1])) * sin(
        x[1]) * sin(x[2]) * sin(x[3]) / x[1] ** 2 + 2 * x[5] * (1 - cos(x[1])) ** 2 * sin(x[2]) * sin(x[3]) / x[1] ** 3,
    (1 - cos(x[0])) ** 2 * cos(x[2]) / x[0] ** 2,
    ((1 - cos(x[1])) ** 2 * cos(x[0]) * cos(x[3]) / x[1] ** 2 + (1 - cos(x[1])) * sin(x[0]) * sin(x[1]) / x[1] ** 2) * cos(x[2]) - (
                1 - cos(x[1])) ** 2 * sin(x[2]) * sin(x[3]) / x[1] ** 2],
            [(-x[5] * (1 - cos(x[1])) ** 2 * sin(x[0]) * cos(x[3]) / x[1] ** 2 + x[5] * (1 - cos(x[1])) * sin(x[1]) * cos(
        x[0]) / x[1] ** 2 + 2 * x[4] * (1 - cos(x[0])) * sin(x[0]) / x[0] ** 2 - 2 * x[4] * (1 - cos(x[0])) ** 2 / x[0] ** 3) * sin(x[2]),
    (x[5] * (1 - cos(x[1])) ** 2 * cos(x[0]) * cos(x[3]) / x[1] ** 2 + x[5] * (1 - cos(x[1])) * sin(x[0]) * sin(
        x[1]) / x[1] ** 2 + x[4] * (1 - cos(x[0])) ** 2 / x[0] ** 2) * cos(x[2]) - x[5] * (1 - cos(x[1])) ** 2 * sin(x[2]) * sin(
        x[3]) / x[1] ** 2,
    (x[5] * (1 - cos(x[1])) * sin(x[0]) * cos(x[1]) / x[1] ** 2 + 2 * x[5] * (1 - cos(x[1])) * sin(x[1]) * cos(x[0]) * cos(
        x[3]) / x[1] ** 2 + x[5] * sin(x[0]) * sin(x[1]) ** 2 / x[1] ** 2 - 2 * x[5] * (1 - cos(x[1])) ** 2 * cos(x[0]) * cos(
        x[3]) / x[1] ** 3 - 2 * x[5] * (1 - cos(x[1])) * sin(x[0]) * sin(x[1]) / x[1] ** 3) * sin(x[2]) + 2 * x[5] * (1 - cos(x[1])) * sin(
        x[1]) * sin(x[3]) * cos(x[2]) / x[1] ** 2 - 2 * x[5] * (1 - cos(x[1])) ** 2 * sin(x[3]) * cos(x[2]) / x[1] ** 3,
    -x[5] * (1 - cos(x[1])) ** 2 * sin(x[2]) * sin(x[3]) * cos(x[0]) / x[1] ** 2 + x[5] * (1 - cos(x[1])) ** 2 * cos(x[2]) * cos(
        x[3]) / x[1] ** 2,
    (1 - cos(x[0])) ** 2 * sin(x[2]) / x[0] ** 2,
    ((1 - cos(x[1])) ** 2 * cos(x[0]) * cos(x[3]) / x[1] ** 2 + (1 - cos(x[1])) * sin(x[0]) * sin(x[1]) / x[1] ** 2) * sin(x[2]) + (
                1 - cos(x[1])) ** 2 * sin(x[3]) * cos(x[2]) / x[1] ** 2],
            [-x[5] * (1 - cos(x[1])) ** 2 * cos(x[0]) * cos(x[3]) / x[1] ** 2 - x[5] * (1 - cos(x[1])) * sin(x[0]) * sin(
        x[1]) / x[1] ** 2 + x[4] * (1 - cos(x[0])) * cos(x[0]) / x[0] ** 2 + x[4] * sin(x[0]) ** 2 / x[0] ** 2 - 2 * x[4] * (
                      1 - cos(x[0])) * sin(x[0]) / x[0] ** 3,
    0,
    -2 * x[5] * (1 - cos(x[1])) * sin(x[0]) * sin(x[1]) * cos(x[3]) / x[1] ** 2 + x[5] * (1 - cos(x[1])) * cos(x[0]) * cos(
        x[1]) / x[1] ** 2 + x[5] * sin(x[1]) ** 2 * cos(x[0]) / x[1] ** 2 + 2 * x[5] * (1 - cos(x[1])) ** 2 * sin(x[0]) * cos(
        x[3]) / x[1] ** 3 - 2 * x[5] * (1 - cos(x[1])) * sin(x[1]) * cos(x[0]) / x[1] ** 3,
    x[5] * (1 - cos(x[1])) ** 2 * sin(x[0]) * sin(x[3]) / x[1] ** 2,
    (1 - cos(x[0])) * sin(x[0]) / x[0] ** 2,
    -(1 - cos(x[1])) ** 2 * sin(x[0]) * cos(x[3]) / x[1] ** 2 + (1 - cos(x[1])) * sin(x[1]) * cos(x[0]) / x[1] ** 2],
            [(-sin(x[0]) * sin(x[1]) * cos(x[3]) + cos(x[0]) * cos(x[1])) * cos(x[2]),
    -(sin(x[0]) * cos(x[1]) + sin(x[1]) * cos(x[0]) * cos(x[3])) * sin(x[2]) - sin(x[1]) * sin(x[3]) * cos(x[2]),
    (-sin(x[0]) * sin(x[1]) + cos(x[0]) * cos(x[1]) * cos(x[3])) * cos(x[2]) - sin(x[2]) * sin(x[3]) * cos(x[1]),
    (-sin(x[0]) * sin(x[1]) + cos(x[0]) * cos(x[1]) * cos(x[3])) * cos(x[2]) - sin(x[2]) * sin(x[3]) * cos(x[1]),
    0,
    0],
            [(-sin(x[0]) * sin(x[1]) * cos(x[3]) + cos(x[0]) * cos(x[1])) * sin(x[2]),
    (sin(x[0]) * cos(x[1]) + sin(x[1]) * cos(x[0]) * cos(x[3])) * cos(x[2]) - sin(x[1]) * sin(x[2]) * sin(x[3]),
    (-sin(x[0]) * sin(x[1]) + cos(x[0]) * cos(x[1]) * cos(x[3])) * sin(x[2]) + sin(x[3]) * cos(x[1]) * cos(x[2]),
    -sin(x[1]) * sin(x[2]) * sin(x[3]) * cos(x[0]) + sin(x[1]) * cos(x[2]) * cos(x[3]),
    0,
    0],
            [-sin(x[0]) * cos(x[1]) - sin(x[1]) * cos(x[0]) * cos(x[3]),
    0,
    -sin(x[0]) * cos(x[1]) * cos(x[3]) - sin(x[1]) * cos(x[0]),
    sin(x[0]) * sin(x[1]) * sin(x[3]),
    0,
    0]
        ]).astype('float64')
    x0_rosenbrock = np.array([pi/3, pi/3, pi/3, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0]).astype('float64')
    res = least_squares(test_3_, x0_rosenbrock,
                        bounds=([0,0,0,0,0,0,0,0,0],[2*pi, 2*pi, 2*pi, 2*pi, 2*pi, 2*pi, np.inf, np.inf, np.inf]))
    print('res.x:', res.x)
    print('result:', test_3_rad(res.x))

def rank_for1(a1= pi/2, b1 = pi/2, rad1 = 1):
    a = np.matrix(
        [[rad1*sin(a1)*cos(b1), -rad1*(1 - cos(a1))*sin(b1), (1 - cos(a1))*cos(b1)],
         [rad1*sin(a1)*sin(b1), rad1*(1 - cos(a1))*cos(b1), (1 - cos(a1))*sin(b1)],
         [rad1*cos(a1), 0, sin(a1)],
        [cos(a1) * cos(b1), -sin(a1) * sin(b1), 0],
         [sin(b1) * cos(a1), sin(a1) * cos(b1), 0],
         [-sin(a1), 0, 0]]
    ).astype('float64')
    print(np.linalg.matrix_rank(a))

def rank_for2(a1= pi/2, a2 = pi/2, b1 = pi/2, b2 = pi/2, rad1 = 1, rad2 = 1):
    a = np.matrix([[(rad1 * sin(a1) - rad2 * (1 - cos(a2)) * sin(a1) * cos(b2) + rad2 * sin(a2) * cos(a1)) * cos(b1),
       -rad2 * sin(a2) * sin(b1) * sin(b2) + (rad2 * sin(a1) * cos(a2) + rad2 * sin(a2) * cos(a1) * cos(b2)) * cos(b1),
       -rad2 * (1 - cos(a2)) * sin(b2) * cos(b1) - (
                   rad1 * (1 - cos(a1)) + rad2 * (1 - cos(a2)) * cos(a1) * cos(b2) + rad2 * sin(a1) * sin(a2)) * sin(
           b1), -rad2 * (1 - cos(a2)) * sin(b1) * cos(b2) - rad2 * (1 - cos(a2)) * sin(b2) * cos(a1) * cos(b1),
       (1 - cos(a1)) * cos(b1),
       ((1 - cos(a2)) * cos(a1) * cos(b2) + sin(a1) * sin(a2)) * cos(b1) + (cos(a2) - 1) * sin(b1) * sin(b2)],
      [(rad1 * sin(a1) - rad2 * (1 - cos(a2)) * sin(a1) * cos(b2) + rad2 * sin(a2) * cos(a1)) * sin(b1),
       rad2 * sin(a2) * sin(b2) * cos(b1) + (rad2 * sin(a1) * cos(a2) + rad2 * sin(a2) * cos(a1) * cos(b2)) * sin(b1),
       -rad2 * (1 - cos(a2)) * sin(b1) * sin(b2) + (
                   rad1 * (1 - cos(a1)) + rad2 * (1 - cos(a2)) * cos(a1) * cos(b2) + rad2 * sin(a1) * sin(a2)) * cos(
           b1), -rad2 * (1 - cos(a2)) * sin(b1) * sin(b2) * cos(a1) + rad2 * (1 - cos(a2)) * cos(b1) * cos(b2),
       (1 - cos(a1)) * sin(b1),
       (1 - cos(a2)) * sin(b2) * cos(b1) + ((1 - cos(a2)) * cos(a1) * cos(b2) + sin(a1) * sin(a2)) * sin(b1)],
      [rad1 * cos(a1) - rad2 * (1 - cos(a2)) * cos(a1) * cos(b2) - rad2 * sin(a1) * sin(a2),
       -rad2 * sin(a1) * sin(a2) * cos(b2) + rad2 * cos(a1) * cos(a2), 0, rad2 * (1 - cos(a2)) * sin(a1) * sin(b2),
       sin(a1), (cos(a2) - 1) * sin(a1) * cos(b2) + sin(a2) * cos(a1)],
    [(-sin(a1) * sin(a2) * cos(b2) + cos(a1) * cos(a2)) * cos(b1),
       (-sin(a1) * sin(a2) + cos(a1) * cos(a2) * cos(b2)) * cos(b1) - sin(b1) * sin(b2) * cos(a2),
       -(sin(a1) * cos(a2) + sin(a2) * cos(a1) * cos(b2)) * sin(b1) - sin(a2) * sin(b2) * cos(b1),
       -sin(a2) * sin(b1) * cos(b2) - sin(a2) * sin(b2) * cos(a1) * cos(b1), 0, 0],
      [(-sin(a1) * sin(a2) * cos(b2) + cos(a1) * cos(a2)) * sin(b1),
       (-sin(a1) * sin(a2) + cos(a1) * cos(a2) * cos(b2)) * sin(b1) + sin(b2) * cos(a2) * cos(b1),
       (sin(a1) * cos(a2) + sin(a2) * cos(a1) * cos(b2)) * cos(b1) - sin(a2) * sin(b1) * sin(b2),
       -sin(a2) * sin(b1) * sin(b2) * cos(a1) + sin(a2) * cos(b1) * cos(b2), 0, 0],
      [-sin(a1) * cos(a2) - sin(a2) * cos(a1) * cos(b2), -sin(a1) * cos(a2) * cos(b2) - sin(a2) * cos(a1), 0,
       sin(a1) * sin(a2) * sin(b2), 0, 0]]).astype('float64')
    print(np.linalg.matrix_rank(a))

def rank_for3(a1=pi / 2, a2=pi / 3, a3=pi/3,b1=pi / 2, b2=pi / 3,b3=pi/2, rad1=2, rad2=2, rad3=4):
    a = np.matrix(
        [[(rad1 * sin(a1) - (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * cos(
            b2)) * sin(a1) + (
                        rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(a3) * cos(a2)) * cos(
            a1)) * cos(b1), (
                       (rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(a3) * cos(a2)) * cos(
                   a1) * cos(b2) + (rad2 * cos(a2) - rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) - rad3 * sin(a2) * sin(
                   a3)) * sin(a1)) * cos(b1) - (
                       rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(a3) * cos(a2)) * sin(
            b1) * sin(b2), ((-rad3 * sin(a2) * sin(a3) * cos(b3) + rad3 * cos(a2) * cos(a3)) * sin(a1) + (
                    -rad3 * sin(a3) * sin(b2) * sin(b3) + (
                        rad3 * sin(a2) * cos(a3) + rad3 * sin(a3) * cos(a2) * cos(b3)) * cos(b2)) * cos(a1)) * cos(
            b1) + (-rad3 * sin(a3) * sin(b3) * cos(b2) - (
                    rad3 * sin(a2) * cos(a3) + rad3 * sin(a3) * cos(a2) * cos(b3)) * sin(b2)) * sin(b1), (
                       -rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) - (
                           rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(
                       a3)) * sin(b2)) * cos(b1) - (rad1 * (1 - cos(a1)) + (
                    -rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                        rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(
                    a3)) * cos(b2)) * cos(a1) + (rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(
            a3) * cos(a2)) * sin(a1)) * sin(b1), (rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) - (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * cos(
            b2)) * sin(b1) + (-rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) - (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * sin(
            b2)) * cos(a1) * cos(b1), (rad3 * (1 - cos(a3)) * sin(a1) * sin(a2) * sin(b3) + (
                    -rad3 * (1 - cos(a3)) * sin(b2) * cos(b3) - rad3 * (1 - cos(a3)) * sin(b3) * cos(a2) * cos(
                b2)) * cos(a1)) * cos(b1) + (
                       rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) * cos(a2) - rad3 * (1 - cos(a3)) * cos(b2) * cos(
                   b3)) * sin(b1), (1 - cos(a1)) * cos(b1),
           -(1 - cos(a2)) * sin(b1) * sin(b2) + ((1 - cos(a2)) * cos(a1) * cos(b2) + sin(a1) * sin(a2)) * cos(b1), ((((
                                                                                                                                  1 - cos(
                                                                                                                              a3)) * cos(
                a2) * cos(b3) + sin(a2) * sin(a3)) * cos(b2) + (cos(a3) - 1) * sin(b2) * sin(b3)) * cos(a1) + ((cos(
                a3) - 1) * sin(a2) * cos(b3) + sin(a3) * cos(a2)) * sin(a1)) * cos(b1) + (
                       -(1 - cos(a3)) * sin(b3) * cos(b2) - (
                           (1 - cos(a3)) * cos(a2) * cos(b3) + sin(a2) * sin(a3)) * sin(b2)) * sin(b1)],
         [(rad1 * sin(
            a1) - (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * cos(
            b2)) * sin(a1) + (rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(a3) * cos(
            a2)) * cos(a1)) * sin(b1), ((rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(
            a3) * cos(a2)) * cos(a1) * cos(b2) + (rad2 * cos(a2) - rad3 * (1 - cos(a3)) * cos(a2) * cos(
            b3) - rad3 * sin(a2) * sin(a3)) * sin(a1)) * sin(b1) + (rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(
            a2) * cos(b3) + rad3 * sin(a3) * cos(a2)) * sin(b2) * cos(b1), ((-rad3 * sin(a2) * sin(a3) * cos(
            b3) + rad3 * cos(a2) * cos(a3)) * sin(a1) + (-rad3 * sin(a3) * sin(b2) * sin(b3) + (
                    rad3 * sin(a2) * cos(a3) + rad3 * sin(a3) * cos(a2) * cos(b3)) * cos(b2)) * cos(a1)) * sin(b1) + (
                                                                                                                      rad3 * sin(
                                                                                                                  a3) * sin(
                                                                                                                  b3) * cos(
                                                                                                                  b2) + (
                                                                                                                                  rad3 * sin(
                                                                                                                              a2) * cos(
                                                                                                                              a3) + rad3 * sin(
                                                                                                                              a3) * cos(
                                                                                                                              a2) * cos(
                                                                                                                              b3)) * sin(
                                                                                                                  b2)) * cos(
            b1), -(rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) + (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * sin(
            b2)) * sin(b1) + (rad1 * (1 - cos(a1)) + (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * cos(
            b2)) * cos(a1) + (rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(a3) * cos(
            a2)) * sin(a1)) * cos(b1), (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) + (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * cos(
            b2)) * cos(b1) + (-rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) - (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * sin(
            b2)) * sin(b1) * cos(a1), (rad3 * (1 - cos(a3)) * sin(a1) * sin(a2) * sin(b3) + (
                    -rad3 * (1 - cos(a3)) * sin(b2) * cos(b3) - rad3 * (1 - cos(a3)) * sin(b3) * cos(a2) * cos(
                b2)) * cos(a1)) * sin(b1) + (-rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) * cos(a2) + rad3 * (
                    1 - cos(a3)) * cos(b2) * cos(b3)) * cos(b1), (1 - cos(a1)) * sin(b1), (1 - cos(a2)) * sin(b2) * cos(
            b1) + ((1 - cos(a2)) * cos(a1) * cos(b2) + sin(a1) * sin(a2)) * sin(b1), ((((1 - cos(a3)) * cos(a2) * cos(
            b3) + sin(a2) * sin(a3)) * cos(b2) + (cos(a3) - 1) * sin(b2) * sin(b3)) * cos(a1) + ((cos(a3) - 1) * sin(
            a2) * cos(b3) + sin(a3) * cos(a2)) * sin(a1)) * sin(b1) + ((1 - cos(a3)) * sin(b3) * cos(b2) + (
                    (1 - cos(a3)) * cos(a2) * cos(b3) + sin(a2) * sin(a3)) * sin(b2)) * cos(b1)],
         [rad1 * cos(a1) + (
                    rad3 * (1 - cos(a3)) * sin(b2) * sin(b3) - (
                        rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(
                    a3)) * cos(b2)) * cos(a1) - (rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(
            a3) * cos(a2)) * sin(a1), -(
                    rad2 * sin(a2) - rad3 * (1 - cos(a3)) * sin(a2) * cos(b3) + rad3 * sin(a3) * cos(a2)) * sin(
            a1) * cos(b2) + (rad2 * cos(a2) - rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) - rad3 * sin(a2) * sin(
            a3)) * cos(a1), (-rad3 * sin(a2) * sin(a3) * cos(b3) + rad3 * cos(a2) * cos(a3)) * cos(a1) + (rad3 * sin(
            a3) * sin(b2) * sin(b3) - (rad3 * sin(a2) * cos(a3) + rad3 * sin(a3) * cos(a2) * cos(b3)) * cos(b2)) * sin(
            a1), 0, (rad3 * (1 - cos(a3)) * sin(b3) * cos(b2) + (
                    rad2 * (1 - cos(a2)) + rad3 * (1 - cos(a3)) * cos(a2) * cos(b3) + rad3 * sin(a2) * sin(a3)) * sin(
            b2)) * sin(a1), rad3 * (1 - cos(a3)) * sin(a2) * sin(b3) * cos(a1) + (rad3 * (1 - cos(a3)) * sin(b2) * cos(
            b3) + rad3 * (1 - cos(a3)) * sin(b3) * cos(a2) * cos(b2)) * sin(a1), sin(a1), -(1 - cos(a2)) * sin(
            a1) * cos(b2) + sin(a2) * cos(a1), (-((1 - cos(a3)) * cos(a2) * cos(b3) + sin(a2) * sin(a3)) * cos(b2) - (
                    cos(a3) - 1) * sin(b2) * sin(b3)) * sin(a1) + ((cos(a3) - 1) * sin(a2) * cos(b3) + sin(a3) * cos(
            a2)) * cos(a1)],
         [(-((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * sin(a1) + (
                    -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1)) * cos(b1), (
                       (-sin(a2) * cos(a3) - sin(a3) * cos(a2) * cos(b3)) * sin(a1) + (
                           -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1) * cos(b2)) * cos(b1) - (
                       -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(b1) * sin(b2), (((-sin(a2) * sin(
            a3) + cos(a2) * cos(a3) * cos(b3)) * cos(b2) - sin(b2) * sin(b3) * cos(a3)) * cos(a1) + (-sin(a2) * cos(
            a3) * cos(b3) - sin(a3) * cos(a2)) * sin(a1)) * cos(b1) + (
                       -(-sin(a2) * sin(a3) + cos(a2) * cos(a3) * cos(b3)) * sin(b2) - sin(b3) * cos(a3) * cos(
                   b2)) * sin(b1), -(
                    ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * cos(
                a1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1)) * sin(b1) + (
                       -(sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) - sin(a3) * sin(b3) * cos(
                   b2)) * cos(b1),
           (-(sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) - sin(a3) * sin(b3) * cos(b2)) * cos(a1) * cos(
               b1) + (-(sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) + sin(a3) * sin(b2) * sin(b3)) * sin(
               b1), ((-sin(a3) * sin(b2) * cos(b3) - sin(a3) * sin(b3) * cos(a2) * cos(b2)) * cos(a1) + sin(a1) * sin(
                a2) * sin(a3) * sin(b3)) * cos(b1) + (
                       sin(a3) * sin(b2) * sin(b3) * cos(a2) - sin(a3) * cos(b2) * cos(b3)) * sin(b1), 0, 0, 0],
         [(-(
                    (sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * sin(
            a1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1)) * sin(b1), ((-sin(a2) * cos(a3) - sin(
            a3) * cos(a2) * cos(b3)) * sin(a1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1) * cos(
            b2)) * sin(b1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(b2) * cos(b1), (((-sin(a2) * sin(
            a3) + cos(a2) * cos(a3) * cos(b3)) * cos(b2) - sin(b2) * sin(b3) * cos(a3)) * cos(a1) + (-sin(a2) * cos(
            a3) * cos(b3) - sin(a3) * cos(a2)) * sin(a1)) * sin(b1) + ((-sin(a2) * sin(a3) + cos(a2) * cos(a3) * cos(
            b3)) * sin(b2) + sin(b3) * cos(a3) * cos(b2)) * cos(b1), (((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(
            b3)) * cos(b2) - sin(a3) * sin(b2) * sin(b3)) * cos(a1) + (-sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(
            a3)) * sin(a1)) * cos(b1) - ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(a3) * sin(
            b3) * cos(b2)) * sin(b1), (-(sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) - sin(a3) * sin(
            b3) * cos(b2)) * sin(b1) * cos(a1) + ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) - sin(
            a3) * sin(b2) * sin(b3)) * cos(b1), ((-sin(a3) * sin(b2) * cos(b3) - sin(a3) * sin(b3) * cos(a2) * cos(
            b2)) * cos(a1) + sin(a1) * sin(a2) * sin(a3) * sin(b3)) * sin(b1) + (-sin(a3) * sin(b2) * sin(b3) * cos(
            a2) + sin(a3) * cos(b2) * cos(b3)) * cos(b1), 0, 0, 0],
         [
              (-(sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * cos(b2) + sin(a3) * sin(b2) * sin(b3)) * cos(a1) - (
                          -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1),
              (-sin(a2) * cos(a3) - sin(a3) * cos(a2) * cos(b3)) * cos(a1) - (
                          -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * sin(a1) * cos(b2),
              (-(-sin(a2) * sin(a3) + cos(a2) * cos(a3) * cos(b3)) * cos(b2) + sin(b2) * sin(b3) * cos(a3)) * sin(
                  a1) + (-sin(a2) * cos(a3) * cos(b3) - sin(a3) * cos(a2)) * cos(a1), 0,
              ((sin(a2) * cos(a3) + sin(a3) * cos(a2) * cos(b3)) * sin(b2) + sin(a3) * sin(b3) * cos(b2)) * sin(a1),
              (sin(a3) * sin(b2) * cos(b3) + sin(a3) * sin(b3) * cos(a2) * cos(b2)) * sin(a1) + sin(a2) * sin(a3) * sin(
                  b3) * cos(a1), 0, 0, 0]]
        ).astype('float64')
    print(np.linalg.matrix_rank(a))

if 0:
    now = time.time()
    x = np.array([pi/2, pi/2, 0, 0, 4, 4], dtype='float')
    x1 = np.array([pi/8, pi/8, pi/3, pi/5, pi/6, pi/5, 4], dtype='float')

    '''result = fsolve(position_3, x)
    print("===================")
    print()
    print("求解函数名称:", fsolve.__name__)
    print("解：", result)
    print("各向量值：", position_3_origin(result))

    last = time.time() - now
    print('\n{0:.4f}s'.format(last))
    now = time.time()'''

    # 拟合函数来求解
    h = leastsq(test_2, x)
    print("===================")
    print()
    print("求解函数名称:", leastsq.__name__)
    print("解：", h[0])
    print("各向量值：", test_2_origin(h[0]))

    last = time.time() - now
    print('\n{0:.4f}s'.format(last))

if 0:
    a1, a2, a3, b1, b2, b3, l1, l2, l3 = symbols('a1 a2 a3 b1 b2 b3 l1 l2 l3')
    trans_b1 = np.array([
        [cos(b1),-sin(b1),0],
        [sin(b1),cos(b1),0],
        [0,0,1]
    ])
    trans_a1 = np.array([
            [cos(a1),0,sin(a1)],
            [0,1,0],
            [-sin(a1),0,cos(a1)]
        ])
    trans_a2 = np.array([
        [cos(a2), 0, sin(a2)],
        [0, 1, 0],
        [-sin(a2), 0, cos(a2)]
    ])
    trans_b2= np.array([
        [cos(b2),-sin(b2),0],
        [sin(b2),cos(b2),0],
        [0,0,1]
    ])
    trans_b3 = np.array([
        [cos(b3), -sin(b3), 0],
        [sin(b3), cos(b3), 0],
        [0, 0, 1]
    ])
    trans_add = np.array([
        [cos(pi/2), -sin(pi/2), 0],
        [sin(pi/2), cos(pi/2), 0],
        [0, 0, 1]
    ])
    lm1 =np.array(
        [[(l1 / a1) * (1 - cos(a1))], [0], [(l1 / a1) * sin(a1)]]
    )
    lm2 = np.array(
        [[(l2 / a2) * (1 - cos(a2))], [0], [(l2 / a2) * sin(a2)]]
    )
    lm3 = np.array(
        [[(l3 / a3) * (1 - cos(a3))], [0], [(l3 / a3) * sin(a3)]]
    )
    n2 = np.array(
        [[cos(b2)*sin(a2)], [sin(b2)*sin(a2)], [cos(a2)]]
    )

    f = trans_b1.dot(trans_a1.dot(trans_b2).dot(lm2)+lm1)
    f1 = trans_b1.dot(trans_a1).dot(n2)

    f = trans_b2.dot(trans_a2.dot(trans_b3).dot(lm3)+f)
    f1 = trans_b2.dot(trans_a2).dot(f1)

    #seg3 带beta旋转
    seg3 = trans_b3.dot(lm3)
    angle3 = n2
    #seg 2+3
    seg2 = trans_b2.dot(trans_a2.dot(seg3)+lm2)
    angle2 = trans_b2.dot(trans_a2).dot(angle3)
    #seg 1+2+3
    seg1 = trans_b1.dot(trans_a1.dot(seg2)+lm1)
    angle1 = trans_b1.dot(trans_a1).dot(angle2)

    seg1 = [(lm2 * (1 - cos(a2)) * cos(a1) * cos(b2) / a2 + lm2 * sin(a1) * sin(a2) / a2 + lm1 * (1 - cos(a1)) / a1) * cos(
        b1) - lm2 * (1 - cos(a2)) * sin(b1) * sin(b2) / a2,
     (lm2 * (1 - cos(a2)) * cos(a1) * cos(b2) / a2 + lm2 * sin(a1) * sin(a2) / a2 + lm1 * (1 - cos(a1)) / a1) * sin(
         b1) + lm2 * (1 - cos(a2)) * sin(b2) * cos(b1) / a2,
     -lm2 * (1 - cos(a2)) * sin(a1) * cos(b2) / a2 + lm2 * sin(a2) * cos(a1) / a2 + lm1 * sin(a1) / a1]
    angle1 = [
     (sin(a1) * cos(a2) + sin(a2) * cos(a1) * cos(b2)) * cos(b1) - sin(a2) * sin(b1) * sin(b2),
     (sin(a1) * cos(a2) + sin(a2) * cos(a1) * cos(b2)) * sin(b1) + sin(a2) * sin(b2) * cos(b1),
     -sin(a1) * sin(a2) * cos(b2) + cos(a1) * cos(a2)]

    print('f1 =', seg1[0][0])
    print('f2 =', seg1[1][0])
    print('f3 =', seg1[2][0])
    print('f4 =', angle1[0])
    print('f5 =', angle1[1])
    print('f6 =', angle1[2])
    x = np.array([pi/6, pi/6, pi/6, 0, 0, 0, 4, 4, 4], dtype='float')
    print(position_3_origin(x))
    '''print('f4 =', f2[0])
    print('f5 =', f2[1])
    print('f6 =', f2[2])'''

    print('\nf11 =',diff(seg1[0],a1)[0])
    print('f12 =',diff(seg1[0],b1)[0])
    print('f13 =',diff(seg1[0],a2)[0])
    print('f14 =',diff(seg1[0],a2)[0])
    #print('f15 =',diff(seg1[0],a3)[0])
    #print('f16 =',diff(seg1[0],b3)[0])
    print('f17 =',diff(seg1[0],l1)[0])
    print('f18 =',diff(seg1[0],l2)[0])
    #print('f19 =',diff(seg1[0],l3)[0])


    print('\nf21 =', diff(seg1[1],a1)[0])
    print('f22 =', diff(seg1[1],b1)[0])
    print('f23 =', diff(seg1[1],a2)[0])
    print('f24 =', diff(seg1[1],b2)[0])
   # print('f25 =', diff(seg1[1],a3)[0])
   # print('f26 =', diff(seg1[1],b3)[0])
    print('f27 =', diff(seg1[1],l1)[0])
    print('f28 =', diff(seg1[1],l2)[0])
    #print('f29 =', diff(seg1[1],l3)[0])

    print('\nf31 =', diff(seg1[2],a1)[0])
    print('f32 =', diff(seg1[2],b1)[0])
    print('f33 =', diff(seg1[2],a2)[0])
    print('f34 =', diff(seg1[2],b2)[0])
  #  print('f35 =', diff(seg1[2],a3)[0])
  #  print('f36 =', diff(seg1[2],b3)[0])
    print('f37 =', diff(seg1[2],l1)[0])
    print('f38 =', diff(seg1[2],l2)[0])
   # print('f39 =', diff(seg1[2],l3)[0])

    print('\nf41 =', diff(angle1[0], a1))
    print('f42 =', diff(angle1[0], b1))
    print('f43 =', diff(angle1[0], a2))
    print('f44 =', diff(angle1[0], a2))
    #print('f45 =', diff(angle1[0], a3))
    #print('f46 =', diff(angle1[0], b3))
    print('f47 =', diff(angle1[0], l1))
    print('f48 =', diff(angle1[0], l2))
    #print('f49 =', diff(angle1[0], l3))

    print('\nf51 =', diff(angle1[1], a1))
    print('f52 =', diff(angle1[1], b1))
    print('f53 =', diff(angle1[1], a2))
    print('f54 =', diff(angle1[1], b2))
  #  print('f55 =', diff(angle1[1], a3))
  #  print('f56 =', diff(angle1[1], b3))
    print('f57 =', diff(angle1[1], l1))
    print('f58 =', diff(angle1[1], l2))
  #  print('f59 =', diff(angle1[1], l3))

    print('\nf61 =', diff(angle1[2], a1))
    print('f62 =', diff(angle1[2], b1))
    print('f63 =', diff(angle1[2], a2))
    print('f64 =', diff(angle1[2], b2))
  #  print('f65 =', diff(angle1[2], a3))
  #  print('f66 =', diff(angle1[2], b3))
    print('f67 =', diff(angle1[2], l1))
    print('f68 =', diff(angle1[2], l2))
  #  print('f69 =', diff(angle1[2], l3))

if 0:
    #derivation(3,1)
    '''res_rad= [1.20393946, 0.93787404, 0.61304159, 0.34791947, 1.60855949, 2.67193574,
              4.83245106, 6.20313642, 9.48964311]
    result= [1.2039394560838992, 0.9378740388363163, 0.6130415866720027, 0.347919466311137, 1.60855948767326,
             2.671935744659466, 5.817978506388819, 5.8177606111290725, 5.817545870442461]'''
    res_rad = [pi/4,pi/4,pi/4,pi/4,pi/4,pi/4,20/pi,20/pi,20/pi]
    result = [pi/4,pi/4,pi/4,pi/4,pi/4,pi/4,5,5,5]
    test_square()
    print(test_3_rad([1.00365353, 1.00073273 ,0.65340687, 0.271493  , 1.30583803 ,2.79513611,
 5.16230662, 5.17744241, 7.92956675]))
    print(test_3_rad([0.95130287, 0.99206756, 0.66422903 ,0.29461994, 1.19270756, 2.82771218 ,5.45156515 ,5.22752532 ,7.80757701]))

if 1:
    rank_for1()

