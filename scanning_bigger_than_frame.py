import numpy as np
import serial
import time

import cv2
import cv2.cv as cv


def deskew(src, pixel_shift):
    src_tri = np.zeros((3, 2), dtype=np.float32)
    dst_tri = np.zeros((3, 2), dtype=np.float32)

    rows, cols, ch = src.shape

    # Set your 3 points to calculate the  Affine Transform
    src_tri[1] = [cols - 1, 0]
    src_tri[2] = [0, rows - 1]
    # dstTri is the same except the bottom is moved over shiftpixels:
    dst_tri[1] = src_tri[1]
    dst_tri[2] = [pixel_shift, rows - 1]

    # Get the Affine Transform
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)

    ## Apply the Affine Transform just found to the src image
    cv2.warpAffine(src, warp_mat, (cols, rows), src, cv2.INTER_CUBIC)
    return src


ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200)
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

count = 0
coin_count = 0
center_list = []
found_coin = False
while (True):
    # Capture frame-by-frame
    cv.WaitKey(2)
    start_time = time.time()
    ret, frame = cap.read()
    # deskewed = deskew(frame, 5)
    print '1 In %s seconds' % (time.time() - start_time,)
    if frame == None:
        # print 'None in %s seconds' % (time.time() - start_time,)
        continue
    # cv2.imwrite('/home/pkrush/cents-hd/' + str(coin_count).zfill(5) + str(count).zfill(2) + '.png', frame)
    print '2 In %s seconds' % (time.time() - start_time,)
    # frame = frame[460:,40:1040]

    coin_size_adjustment_factor = 1.0
    frame_width = int(960 * coin_size_adjustment_factor)
    frame_hieght = int(540 * coin_size_adjustment_factor)
    print '3 In %s seconds' % (time.time() - start_time,)
    frame = cv2.resize(frame, (frame_width, frame_hieght), interpolation=cv2.INTER_AREA)
    print '4 In %s seconds' % (time.time() - start_time,)

    cv2.imshow('frame', frame)
    # frame = deskew(frame,-9)  #100 rpm
    # frame = deskew(frame, -30)
    print '5 In %s seconds' % (time.time() - start_time,)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 300, param1=50, param2=30, minRadius=448, maxRadius=450)

    # good for scanning:
    # circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 300, param1=40, param2=20, minRadius=222, maxRadius=226)
    print '6 In %s seconds' % (time.time() - start_time,)

    circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 450, param1=40, param2=20, minRadius=460, maxRadius=480)
    print '7 In %s seconds' % (time.time() - start_time,)

    if circles is None:
        continue

    circles = np.uint16(np.around(circles))

    print 'circles:', len(circles)

    for i in circles[0, :]:
        center_x = i[0]
        center_y = i[1]
        crop_radius = i[2]
        # print circles
        center_stop = 250
        print center_x, "   ", center_y, "     ", crop_radius
        if center_x > center_stop:
            print center_x
            if found_coin == False:
                found_coin = True
                print "Conveyor Stop"
                ser.write(str(103) + "\n")
                cv.WaitKey(2)
                continue
            print count
            led = count / 2
            if led < 18:
                print "led", led
                ser.write(str(led) + "\n")
                cv.WaitKey(2)

            center_list.append([center_x, center_y, crop_radius])

            sample_size = 10

            total_center_x = 0
            total_center_y = 0
            total_radius = 0

            for past_x, past_y, past_radius in center_list[len(center_list) - sample_size:len(center_list)]:
                total_center_x += past_x
                total_center_y += past_y
                total_radius += past_radius

            average_center_x = float(total_center_x) / sample_size
            average_center_y = float(total_center_y) / sample_size
            average_radius = float(total_radius) / sample_size

            print average_center_x, center_x, "   ", average_center_y, center_y, "     ", average_radius, crop_radius
            print center_x, "   ", center_y, "     ", crop_radius

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(frame, (int(average_center_x), int(average_center_y)), int(average_radius), (0, 255, 0), 1)
            cv2.circle(frame, (int(average_center_x), int(average_center_y)), 2, (0, 0, 255), 1)
            cv2.imshow('detected circles', frame)

            # cv2.circle(frame, (center_x, center_y), crop_radius, (0, 255, 0), 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print '8 In %s seconds' % (time.time() - start_time,)

ser.write(str(102) + "\n")
cv.WaitKey(3500)
ser.write(str(100) + "\n")
cv.WaitKey(100)
ser.write(str(101) + "\n")

cap.release()
cv2.destroyAllWindows()
