import threading
import time
import sys
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
import struct
from PyQt5 import QtGui, QtCore, QtWidgets


class sendback():
    def __init__(self,pre):
        self.serialport = serial.Serial()
        self.serialport.port = 'COM2'
        self.serialport.baudrate = '115200'
        self.serialport.timeout = 0.5
        self.pressure = [.0]*6
        self.serialport.open()
        self.pre = pre
        self.recv = [.0]*6
        self.enable()

    def enable(self):
        self.receiver_thread = threading.Thread(target=self.reader, name='rec')
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

        self.writer_thread = threading.Thread(target=self.writer, name='wrt')
        self.writer_thread.daemon = True
        self.writer_thread.start()

    def writer(self):
        while 1:
            self.serialport.write(self.pre)
            self.serialport.flush()
            time.sleep(1)

    def reader(self):
        while 1:
            data = self.serialport.readline()
            if data != b'':
                data = data.decode()
                index = []
                j = 0
                for i in range(6):
                    index.append(data.find('s', j))
                    index.append(data.find('s', j + 1))
                    # print(j)
                    # print(result[j])
                    length = int(data[index[2 * i] + 1:index[2 * i + 1]])
                    j += int(length) + 2 + len(str(length))
                    self.recv[i]=float(data[index[2 * i + 1] + 1:index[2 * i + 1] + length + 1])
                #print(self.recv)
                self.pre = test_write(self.recv)

def test_read():
    index = list()
    j = 0
    for i in range(6):
        index.append(result.find('s',j))
        index.append(result.find('s',j+1))
        #print(j)
        #print(result[j])
        length = int(result[index[2*i]+1:index[2*i+1]])
        j += int(length)+2+len(str(length))
        print(result[index[2*i+1]+1:index[2*i+1]+length+1])
    print(index)

def test_write(pre_data):
    separate = b's'
    eol = b'\r\n'
    byte_data = bytes()

    for i in range(6):
        lens = str(len(str(pre_data[i])))
        byte_data += separate + lens.encode() + separate + str(pre_data[i]).encode()
    byte_data += eol
    return byte_data

if __name__ == '__main__':
    ports = list()
    for n, (port, desc, hwid) in enumerate(sorted(comports()), 1):
        sys.stderr.write('--- {:2}: {:20} {!r}\n'.format(n, port, desc))
        ports.append(port)

    pre = [123.3,313.43,34523.1,343215.34,56457.2,100.1]
    #pre = [1.01]*6
    send = str(pre[0])+str(pre[1]+1.0)+str(pre[2]+2)+str(pre[3]+3)+str(pre[4]+4)+str(pre[5]+5)+'\r\n'

    #writer
    byte_data = test_write(pre)
    print(byte_data)
    #writer end

    #reader
    'test_write(result)'
    #reader end

    ser = sendback(byte_data)
    print(ser)
    while 1:
        pre


