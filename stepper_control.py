import time
import serial
import sys
import cv2

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(port='/dev/ttyUSB1', baudrate=9600)


print ser.isOpen()
print cv2.__version__
time.sleep(2)

for input in range(0,16):
    out = ''
    while ser.inWaiting() > 0:
        out += ser.read(1)
    if out != '':
        print 'From' + out
    ser.write(str(input) + '\n')
    time.sleep(1)
    #print input
ser.close()
print sys.version