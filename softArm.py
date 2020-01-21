# -*- coding: utf-8 -*-

from math import sin, cos, pi, atan2, sqrt, asin, atan
import numpy


class softArm(object):
    def __init__(self, c1 =6 * 400 * 0.09 / 2.3e-3 / 2, lengthD=0.403,length=0.403
                 ,length0=0.403,alphaD=.0,betaD=.0,alpha=.0,beta=.0,actuator_type='small'):
        self.initialBellowConfigurationAngle = [.0, pi/3, 2*pi/3, pi, -2*pi/3 ,-pi/3]

        self.bellowConfigurationAngleCos = numpy.array([.0]*6)
        self.bellowConfigurationAngleSin = numpy.array([.0]*6)

        self.pressureD = [.0]*6
        self.pressure  = [.0]*6
        self.pressureLimit_Upper = 210  #Kpa
        self.pressureLimit_Lower = 70
        if actuator_type == 'big':
            self.c1 = c1
            self.radR = 0.09
        elif actuator_type == 'small':
            self.c1 = c1
            self.radR = 0.0615

        self.angleULimit = pi/2
        self.angleDLimit = 0
        self.lengthS = 0
        self.posVector = numpy.array([.0, .0, .0])
        self.posVector_D = numpy.array([.0, .0, .0])
        self.speedVector = numpy.array([.0, .0, .0])
        self.angleVelocity = numpy.array([.0, .0, .0])

        self.lengthD = lengthD
        self.length = length
        self.length0 = length0

        self.alphaD = alphaD
        self.betaD = betaD
        self.alpha = alpha
        self.beta = beta
        self.flag = 0

        for i in range(6):
            self.bellowConfigurationAngleCos[i] = cos(self.initialBellowConfigurationAngle[i])
            self.bellowConfigurationAngleSin[i] = sin(self.initialBellowConfigurationAngle[i])

        self.ABLD2PD()

    def constriant(self):
        1

    def ABLD2PD(self):
        b1 = 2 * self.c1 * (self.lengthD - self.length0) / self.radR / 6
        btemp = self.c1 * self.alphaD / 6
        b2 = btemp * cos(self.betaD)
        b3 = 1.7320508 * btemp * sin(self.betaD)
        self.pressureD[0] = b1 + b2 * 2
        self.pressureD[1] = b1 + b2 + b3
        self.pressureD[2] = b1 - b2 + b3
        self.pressureD[3] = b1 - b2 * 2
        self.pressureD[4] = b1 - b2 - b3
        self.pressureD[5] = b1 + b2 - b3
        return self.pressureD

    def PD2ABLD(self):
        phycD = numpy.dot(self.pressureD,self.bellowConfigurationAngleCos)
        physD = numpy.dot(self.pressureD,self.bellowConfigurationAngleSin)
        self.alphaD = sqrt(phycD**2+physD**2)/self.c1
        self.betaD = atan2(physD,phycD)
        self.lengthD = (self.pressureD[0]+self.pressureD[1]+self.pressureD[2]+self.pressureD[3]+
                   self.pressureD[4]+self.pressureD[5])/self.c1/2*self.radR+self.length0
        return self.alphaD, self.betaD, self.lengthD

    def ABL2P(self):
        b1 = 2 * self.c1 * (self.length - self.length0) / self.radR / 6
        btemp = self.c1 * self.alpha / 6
        b2 = btemp * cos(self.beta)
        b3 = 1.7320508 * btemp * sin(self.beta)
        self.pressure[0] = b1 + b2 * 2
        self.pressure[1] = b1 + b2 + b3
        self.pressure[2] = b1 - b2 + b3
        self.pressure[3] = b1 - b2 * 2
        self.pressure[4] = b1 - b2 - b3
        self.pressure[5] = b1 + b2 - b3
        return self.pressure

    def P2ABL(self):
        phyc = numpy.dot(self.pressure,self.bellowConfigurationAngleCos)
        phys = numpy.dot(self.pressure,self.bellowConfigurationAngleSin)
        self.alpha = sqrt(phyc**2+phys**2)/self.c1
        self.beta = atan2(phys,phyc)
        self.length = (self.pressure[0]+self.pressure[1]+self.pressure[2]+self.pressure[3]+
                   self.pressure[4]+self.pressure[5])/self.c1/2*self.radR+self.length0
        return self.alpha, self.beta, self.length

    def Pos2ABLD(self):
        x = self.posVector[0]
        y = self.posVector[1]
        z = self.posVector[2]
        mod = x**2 + y**2
        self.flag += 1
        #print('Pos2ABLD',x,y,z,'flag',self.flag)
        if mod <= z**2:
            self.alphaD = asin(2*sqrt(mod)*z/(z**2+mod))
            #print(self.alphaD)
            if x == .0: #根据x的正负性质 确定beta atan输出值域(-90,90)
                if y > .0:
                    self.betaD = -pi/2
                else:
                    self.betaD = pi/2
            elif x > 0:
                self.betaD = atan(y/x)
            elif x < 0:
                self.betaD = atan(y/x) + pi
            self.lengthD = (z**2+mod)/(2*sqrt(mod))*self.alphaD
            #print(self.lengthD)
        elif mod >z**2:
            self.betaD = atan(y / x)
        return self.alphaD, self.betaD, self.lengthD

    def P2P(self):
        self.speedVector = self.posVector_D - self.posVector_now

    def Acturate(self, mode = 1):
        if mode == 0: #ABL mode
            self.alphaD += self.angleVelocity[0]
            self.betaD += self.angleVelocity[1]
            self.lengthD += self.angleVelocity[2]
            if self.alphaD >= pi/2:
                #self.alphaD = pi/2
                1
            if self.alphaD <= 0:
                self.alphaD = 0
        elif mode: #vector mode
            self.posVector = numpy.add(self.posVector, self.speedVector)
            #print('before',self.posVector)
            x = self.posVector[0]
            y = self.posVector[1]
            z = self.posVector[2]
            #print('Acturate',self.posVector)
            if x**2+y**2>z**2: #限制弯曲不超过上限
                self.posVector = numpy.subtract(self.posVector, self.speedVector)
            self.Pos2ABLD()
            #print('last',self.posVector)

    def SetW(self, alphaW=None, betaW=None, length=None): #设定alpha beta角速度
        if alphaW != None:
             self.angleVelocity[0] = alphaW
        if betaW != None:
            self.angleVelocity[1] = betaW
        if length != None:
            self.angleVelocity[2] = length

    def SetV(self, Vx=None, Vy=None, Vz=None): #设定绝对速度 仅改变输入向量 无输入则不变
        if Vx != None:
            self.speedVector[0] = Vx
        if Vy != None:
            self.speedVector[1] = Vy
        if Vz != None:
            self.speedVector[2] = Vz

    def SetP(self, X=None, Y=None, Z=None):
        if X != None:
            self.posVector[0] = X
        if Y != None:
            self.posVector[1] = Y
        if Z != None:
            self.posVector[2] = Z

    def SetPara(self, A=None, B=None, L=None):
        if A != None:
            self.alphaD = A
        if B != None:
            self.betaD = B
        if L != None:
            self.lengthD = L

    def GetABL(self, mode=0): #供animated3Dplot 使用获取ABL后转换为压力再重新转换为ABL
        self.Acturate(mode)
        self.ABLD2PD()
        self.PD2ABLD()
        return self.alphaD, self.betaD, self.lengthD

if __name__ == '__main__':
    #soft = softArm(alphaD=pi,betaD=pi,lengthD=20)
    #soft.SetV(5,5,5)
    #pre = soft.ABLD2PD()
    a = [0,1,3]
    print(a[-1])
