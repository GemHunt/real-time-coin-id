#This is just used to see the webcam output

import time

import cv2

cap = cv2.VideoCapture(1)
cap.set(3,1920)
cap.set(4,1080)

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
    # coin_size_adjustment_factor = 1.8
    # frame_width = int(960 * coin_size_adjustment_factor)
    # frame_hieght = int(540 * coin_size_adjustment_factor)
    # frame = cv2.resize(frame, (frame_width, frame_hieght), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #red = frame[:, :, 2]
    #green = frame[:, :, 1]
    #blue = frame[:, :, 0]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print 'In %s seconds' % (time.time() - start_time,)

cap.release()
cv2.destroyAllWindows()
