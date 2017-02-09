import numpy as np
import cv2
import cv2.cv as cv
import time

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
cap.set(3,1920)
cap.set(4,1080)

count = 0
coin_count = 306
while (True):
    # Capture frame-by-frame
    cv.WaitKey(2)
    start_time = time.time()
    ret, frame = cap.read()
    #deskewed = deskew(frame, 5)
    if frame == None:
        #print 'None in %s seconds' % (time.time() - start_time,)
        continue
    #cv2.imwrite('/home/pkrush/cents-hd/' + str(coin_count).zfill(5) + str(count).zfill(2) + '.png', frame)
    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)
    frame = deskew(frame,-9)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 300, param1=50, param2=30, minRadius=448, maxRadius=450)
    circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 300, param1=40, param2=20, minRadius=222, maxRadius=226)

    if circles is None:
        continue

    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        center_x = i[0]
        center_y = i[1]
        crop_radius = i[2]
        # cv2.circle(frame, (center_x, center_y), crop_radius, (0, 255, 0), 1)
        # cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 1)
        print circles
        cv2.imshow('detected circles', frame)
        center_start_x = 330
        center_end_x = 670
        if count > 10 and center_x > center_end_x:
            count = 0
            coin_count += 1

        if center_start_x < center_x < center_end_x:
            print count
            crop = frame[center_y - 224:center_y + 224, center_x - 224:center_x + 224]
            cv2.imshow('crop', crop)
            filename = '/home/pkrush/cents/' + str(coin_count) + str(count).zfill(2) + '.png'
            print filename
            #cv2.imwrite(filename, crop)
            count += 1

    #red = frame[:, :, 2]
    #green = frame[:, :, 1]
    #blue = frame[:, :, 0]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print 'In %s seconds' % (time.time() - start_time,)

cap.release()
cv2.destroyAllWindows()
