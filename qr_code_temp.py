#This is just used to see the webcam output

import time
import cv2
import zbar
import Image
import numpy as np
import pyqrcode

background_width = 1280
background_height = 720
padding = 10
qr_scale = 4
qr_unscaled_size = 21  # version 1
qr_size = qr_unscaled_size * qr_scale
x_qr_interval = background_width - (padding * 2) - qr_size
y_qr_interval = background_height - (padding * 2) - qr_size


def get_qr_code(data,scale):
    qr = pyqrcode.create(data, error='H', version=1, mode='binary')
    qr_code = np.array(qr.code, dtype=np.uint8)
    qr_code[qr_code == 0] = 255
    qr_code[qr_code == 1] = 0
    qr_code_size = qr_code.shape[0] * scale
    qr_code = cv2.resize(qr_code, (qr_code_size, qr_code_size), interpolation=cv2.INTER_AREA)
    return qr_code

def display_background():
    background = np.zeros((background_height,background_width), dtype=np.uint8)
    background = background + 255

    for x in range(0,2):
        for y in range(0, 2):
            data = str(x) + ',' + str(y)
            qr_code = get_qr_code(data, qr_scale)
            qr_x = x * x_qr_interval + padding
            qr_y = y * y_qr_interval + padding
            background[qr_y:qr_y + qr_size,qr_x:qr_x + qr_size] = qr_code
    while True:
        cv2.imshow("background", background)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def process_video():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/pkrush/find-parts-faster/0.mp4")
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
            break
        #cv2.imwrite(str(x) + '.png', frame)
        #frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        #coin_size_adjustment_factor = 1.8
        #frame_width = int(960 * coin_size_adjustment_factor)
        #frame_hieght = int(540 * coin_size_adjustment_factor)
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
        print"                                             for symbol in image:" + str(len(image.symbols))

        if len(image.symbols) == 4:
            max_src = np.zeros((4, 2), dtype=np.float32)
            max_dst = np.zeros((4, 2), dtype=np.float32)
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

                    offset_x = 0
                    offset_y = 0

                    if symbol.data == "0,0":
                        max_src[0] = loc[0]
                        qr_code_x = 0
                        qr_code_y = 0

                    if symbol.data == "0,1":
                        max_src[1] = loc[1]
                        qr_code_x = 0
                        qr_code_y = 1

                    if symbol.data == "1,0":
                        max_src[3] = loc[3]
                        qr_code_x = 1
                        qr_code_y = 0

                    if symbol.data == "1,1":
                        max_src[2] = loc[2]
                        qr_code_x = 1
                        qr_code_y = 1

                    local_offset_x = offset_x + x_qr_interval * qr_code_x
                    local_offset_y = offset_y + y_qr_interval * qr_code_y

                    local_dst = np.zeros((4,2), dtype=np.float32)
                    local_dst[0] = [local_offset_x,local_offset_y]
                    local_dst[1] = [local_offset_x, local_offset_y + qr_size]
                    local_dst[2] = [local_offset_x + qr_size, local_offset_y + qr_size]
                    local_dst[3] = [local_offset_x + qr_size, local_offset_y]

                    #dst is all 4 qr_codes:
                    max_offset_x = offset_x + qr_size * 1
                    max_offset_y = offset_y + qr_size * 1
                    max_dst[0] = [offset_x, offset_y]
                    max_dst[1] = [offset_x, max_offset_y + qr_size]
                    max_dst[2] = [max_offset_x + qr_size, max_offset_y + qr_size]
                    max_dst[3] = [max_offset_x + qr_size, offset_y]

                M = cv2.getPerspectiveTransform(max_src, max_dst)
                warped = cv2.warpPerspective(output, M, (960,540))
                cv2.imshow("warped", warped)

        display = cv2.resize(output, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.imshow("Camera", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #print 'In %s seconds' % (time.time() - start_time,)

    cap.release()
    cv2.destroyAllWindows()

#display_background()
process_video()