#This is just used to see the webcam output

import time
import cv2
import zbar
import Image
import numpy as np
cap = cv2.VideoCapture(0)
#cap.set(3,1920)
#cap.set(4,1080)

scanner = zbar.ImageScanner()
scanner.parse_config('enable')

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
    coin_size_adjustment_factor = 1.8
    frame_width = int(960 * coin_size_adjustment_factor)
    frame_hieght = int(540 * coin_size_adjustment_factor)
    # frame = cv2.resize(frame, (frame_width, frame_hieght), interpolation=cv2.INTER_AREA)

    output = frame.copy()

    # raw detection code
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dstCn=0)
    pil = Image.fromarray(gray)
    width, height = pil.size
    raw = pil.tostring()

    # create a reader
    image = zbar.Image(width, height, 'Y800', raw)
    scanner.scan(image)

    # extract results
    #print"                                             for symbol in image:" + str(len(image.symbols))

    if len(image.symbols) == 4:
        for symbol in image:
            # do something useful with results
            if symbol.data in ["0,0","1,0","0,1","1,1"]:
                #print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data
                loc =  symbol.location
                print symbol.data ,loc
                cv2.line(output, loc[0],loc[1], (0, 0, 0))
                cv2.line(output, loc[1], loc[2], (0, 0, 0))
                cv2.line(output, loc[2], loc[3], (0, 0, 0))
                cv2.line(output, loc[3], loc[0], (0, 0, 0))
                src = np.array(loc,dtype = np.float32)
                rad = 3
                #dst = np.float32(((240+rad,240-rad), (240+rad, 320-rad), (320+rad,320-rad), (320+rad,240-rad)))
                if symbol.data == "0,0":
                    dst = np.float32(((160, 50), (160, 110), (225, 110), (225, 50)))
                    M = cv2.getPerspectiveTransform(src,dst)
                    warped = cv2.warpPerspective(output, M, (640,480))
                    cv2.imshow("warped", warped)
                if symbol.data == "1,1":
                    dst = np.float32(((387,322), (387,387), (452, 387), (452, 322)))
                    M = cv2.getPerspectiveTransform(src,dst)
                    warped = cv2.warpPerspective(output, M, (640,480))
                    cv2.imshow("warped", warped)


    # show the frame
    cv2.imshow("#iothack15", output)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #print 'In %s seconds' % (time.time() - start_time,)

cap.release()
cv2.destroyAllWindows()


