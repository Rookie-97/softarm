import numpy as np
from numpy import cos, sin, pi
from scipy.optimize import least_squares

def backward_for_3X3(dst, angle, x0):  # a1 a2 a3 b1 b2 b3 r1 r2 r3
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
                          -sin(a2) * sin(a3) * cos(b3) + cos(a2) * cos(a3)) * cos(a1)]
        return result

    x0_rosenbrock = np.array(x0).astype('float64')
    res = least_squares(test_3, x0_rosenbrock,
                        bounds=(
                        [0, 0, 0, -pi, -pi, -pi, 0.0, 0.0, 0.0], [pi, pi, pi, pi, pi, pi, np.inf, np.inf, np.inf]))

    # print('x:', res.x)
    # print('cost:',res.cost)
    result = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(9):
        result[i] = res.x[i]

    result[6] = result[6] * result[0]
    result[7] = result[7] * result[1]
    result[8] = result[8] * result[2]
    '''print('result:', result)
    print('position:',test_3_rad(res.x))'''
    return result

'''def test_3(x):
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
        return result.astype('float64')'''