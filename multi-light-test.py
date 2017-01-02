# This works right now for a web cam. It would be nice if this was refactored into classes.
# This does not find the center of the coin or resize right now.
# so it's expecting the height of the camera to be adjusted so the penny is 406 pixals in diameter.
# you have to move the penny to get to the correct spot to be cropped out.

import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import cv2.cv as cv
import time

def rotate(img, angle):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cv2.warpAffine(img, M, (cols, rows), img, cv2.INTER_CUBIC)
    return img


def crop_for_date(src):
    dst = src[250:250 + 64, 307:307 + 64]
    dst = cv2.resize(dst, (28, 28), interpolation=cv2.INTER_AREA)
    return dst


def deskew(src, pixel_shift):
    src_tri = np.zeros((3, 2), dtype=np.float32)
    dst_tri = np.zeros((3, 2), dtype=np.float32)

    rows, cols, ch = src.shape

    # Set your 3 points to calculate the  Affine Transform
    src_tri[1] = [cols - 1, 0]
    src_tri[2] = [0, rows - 1]
    # dstTri is the same except the bottom is moved over shiftpixels:
    dst_tri[1] = src_tri[1]
    dst_tri[2] = [pixel_shift, rows-1]

    # Get the Affine Transform
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)

    ## Apply the Affine Transform just found to the src image
    cv2.warpAffine(src, warp_mat, (cols, rows), src, cv2.INTER_CUBIC)
    return src

cap = cv2.VideoCapture(0)
#copper60 = get_classifier("copper60", 60)
#heads_with_rotation64 = get_classifier("heads-with-rotation64", 64)
#dates_over_50 = get_classifier("dates-over-50", 28)
count = 0
while (True):
    # Capture frame-by-frame
    start_time = time.time()
    ret, frame = cap.read()
    #deskewed = deskew(frame, 5)
    print 'Cap %s seconds' % (time.time() - start_time,)
    cv2.imshow('frame', frame)
    print 'Show %s seconds' % (time.time() - start_time,)

    red = frame[:, :, 2]
    green = frame[:, :, 1]
    blue = frame[:, :, 0]

    cv2.imshow('red', red)
    cv2.imshow('green', green)
    cv2.imshow('blue', blue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        time.sleep(1)
    continue

    # Our operations on the frame come here
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    center_x = 0
    center_y = 0
    crop_radius = 0
    cv2.imshow('frame', gray)
    circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 300, param1=50, param2=30, minRadius=196, maxRadius=198)
    if circles is None:
        continue

    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
        cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 1)
        center_x = i[0]
        center_y = i[1]
        crop_radius = i[2]
        print circles
        cv2.imshow('detected circles', frame)

    if center_x < crop_radius:
        continue

    crop_end_x = center_x + crop_radius
    if crop_end_x > gray.shape[1]:
        continue

    # for the microscope camera and cropping to 406 square:
    # gray = gray[27:433, 117:523]
    gray = gray[center_y - crop_radius:center_y + crop_radius, center_x - crop_radius:center_x + crop_radius]
    gray = cv2.resize(gray, (406, 406), interpolation=cv2.INTER_AREA)
    cv2.imshow('crop', gray)

    copper60_score = copper60.predict(get_caffe_image(gray, 60), oversample=False)
    # print copper60_score
    heads_with_rotation64_score = heads_with_rotation64.predict(get_caffe_image(gray, 64), oversample=False)
    # print heads_with_rotation64_score
    count = count + 1
    # print count
    max_value = np.amax(heads_with_rotation64_score)
    angle = np.argmax(heads_with_rotation64_score)
    rotated = rotate(gray, 360 - angle)
    cv2.imshow('rotated', rotated)
    # print max_value,angle
    dateCrop = crop_for_date(rotated)
    cv2.imshow('dateCrop', dateCrop)

    dates_over_50_score = dates_over_50.predict(get_caffe_image(dateCrop, 28), oversample=False)
    # print dates_over_50_score
    date_labels = get_labels("dates-over-50")
    predicted_date = date_labels[np.argmax(dates_over_50_score)]
    # print predicted_date
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rotated, predicted_date, (300, 240), font, 1, (0, 0, 0), 2)

    cv2.imshow('rotated', rotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
