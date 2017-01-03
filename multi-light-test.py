import numpy as np
import cv2
import cv2.cv as cv
import time

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

count = 0
while (True):
    # Capture frame-by-frame
    cv.WaitKey(10)
    start_time = time.time()
    ret, frame = cap.read()
    #deskewed = deskew(frame, 5)
    if frame == None:
        #print 'None in %s seconds' % (time.time() - start_time,)
        continue

    cv2.imshow('frame', frame)

    #red = frame[:, :, 2]
    #green = frame[:, :, 1]
    #blue = frame[:, :, 0]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print 'In %s seconds' % (time.time() - start_time,)

cap.release()
cv2.destroyAllWindows()
