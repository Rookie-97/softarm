from softArm import softArm
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from backward import backward_for_3X3

class SoftObject(object):
    def __init__(self, *Arms):
        self.num = len(Arms)
        self.seg = dict()
        self.pts = dict()
        for i in range(self.num):
            self.seg[i] = Arms[i]

    def getArms(self):
        return self.seg

    def backward_position(self, dst_pos, dst_dir, now):
        #dst_pos = [x, y, z]
        #dst_dir = [alpha, beta, gamma]
        #pos_now = [self.alpha[0], self.alpha[3], self.alpha[6],
                   #self.beta[0], self.beta[3], self.beta[6],
                   #self.lm[0]/self.alpha[0],
                   #self.lm[3]/self.alpha[3],
                   #self.lm[6]/self.alpha[6]]
        desired_args = backward_for_3X3(self.dst_pos, self.dst_dir, now)
        return desired_args

    def path_tracking(self):
        1

    def serial(self):
        1

if __name__ == '__main__':
    SoftObject(2)