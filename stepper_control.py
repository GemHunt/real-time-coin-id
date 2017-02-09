import time
import serial
import sys
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(port='/dev/ttyUSB1', baudrate=9600)

time.sleep(1)

# for input in range(0,16):
#     out = ''
#     while ser.inWaiting() > 0:
#         out += ser.read(1)
#     if out != '':
#         print 'From' + out
#     ser.write(str(input) + '\n')
#     time.sleep(1)
#     #print input
# ser.close()

out = ''

motor0 = 0
motor1 = 0
for x in range(0,400000):
    # Capture frame-by-frame
    start_time = time.time()
    ret, frame = cap.read()
    #deskewed = deskew(frame, 5)
    if frame == None:
        #print 'None in %s seconds' % (time.time() - start_time,)
        continue
    #cv2.imwrite(str(x) + '.png', frame)
    #frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)

    # for input in range(0,3):
    #     input_key = cv2.waitKey(1)
    #     out = ''
    #     while ser.inWaiting() > 0:
    #         out += ser.read(1)
    #     if out != '':
    #         print 'From' + out
    #     ser.write(str(input) + '\n')
    #     time.sleep(.1)
        #print input

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #red = frame[:, :, 2]
    #green = frame[:, :, 1]
    #blue = frame[:, :, 0]



    input_key =cv2.waitKey(1)
    if input_key == ord('l'):
        break



    if input_key == ord('q'):
        motor0 = 3
    if input_key == ord('a'):
        motor0 = 0
    if input_key == ord('z'):
        motor0 = 1

    if input_key == ord('w'):
        motor1 = 4
    if input_key == ord('s'):
        motor1 = 0
    if input_key == ord('x'):
        motor1 = 12

    motor_move = motor0 + motor1

    if input_key != -1:
        while ser.inWaiting() > 0:
            out += ser.read(1)
        if out != '':
            print 'From' + out
        bit8 = str(motor_move) + '\n'
        ser.write(bit8)
        print 'motor_move:', motor_move
        time.sleep(.1)




    #ser.write(str(input) + '\n')
    #time.sleep(1)

    #print 'In %s seconds' % (time.time() - start_time,)

cap.release()
cv2.destroyAllWindows()

ser.close()





print sys.version