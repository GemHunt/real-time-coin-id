import numpy as np
import serial
import time

import cv2
import cv2.cv as cv


def get_work_data():
    return 'data'


def read_from_cameras(top_camera, bottom_camera):
    ret, top = top_camera.read()
    ret, bottom = bottom_camera.read()
    if top == None:
        raise ValueError('A frame from the top camera came up None')
    if bottom == None:
        raise ValueError('A frame from the bottom camera came up None')
    return top, bottom

def deskew(src, pixel_shift):
    src_tri = np.zeros((3, 2), dtype=np.float32)
    dst_tri = np.zeros((3, 2), dtype=np.float32)

    rows = src.shape[0]
    cols = src.shape[1]


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


def scan(top_camera, bottom_camera, ser):
    top_captures = []
    bottom_captures = []

    for count in range(0, 62):
        top, bottom = read_from_cameras(top_camera, bottom_camera)

        if count > 4:
            top_captures.append(top)
            bottom_captures.append(bottom)
        led = count / 2
        if led < 29:
            ser.write(str(led) + "\n")
            cv.WaitKey(1)
    return top_captures, bottom_captures


def save(captures, coin_id):
    count = 0
    center_list = []
    resized = []

    for frame in captures:
        if coin_id % 2 == 0:
            ratio = .41
        else:
            ratio = .46

        frame_width = int(1920 * ratio)
        frame_height = int(1080 * ratio)
        # print '3 In %s seconds' % (time.time() - start_time,)
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        height_expansion_amount = 40
        blank_image = np.zeros((frame_height + height_expansion_amount, frame_width, 3), np.uint8)
        blank_image[height_expansion_amount / 2:frame_height + height_expansion_amount / 2, 0:frame_width] = frame
        frame = blank_image
        resized.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if coin_id % 2 == 0:
            circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 2000, param1=45, param2=25, minRadius=222,
                                       maxRadius=226)
        else:
            circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 2000, param1=45, param2=25, minRadius=222,
                                       maxRadius=226)
        if circles is None:
            continue
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center_x = i[0]
            center_y = i[1]
            crop_radius = i[2]
            cv2.circle(gray, (center_x, center_y), 2, (0, 0, 255), 1)
            cv2.circle(gray, (center_x, center_y), crop_radius, (0, 0, 255), 1)
            center_list.append([center_x, center_y, crop_radius])

    total_center_x = 0
    total_center_y = 0
    total_radius = 0

    for center_x, center_y, crop_radius in center_list:
        print center_x, center_y, crop_radius
        total_center_x += center_x
        total_center_y += center_y
        total_radius += crop_radius

    if len(center_list) == 0:
        raise ValueError(str(coin_id) + 'had no detected circles')

    average_center_x = float(total_center_x) / len(center_list)
    average_center_y = float(total_center_y) / len(center_list)
    average_radius = float(total_radius) / len(center_list)

    print average_center_x, center_x, "   ", average_center_y, center_y, "     ", average_radius, crop_radius

    for frame in resized:
        crop = frame[average_center_y - 224:average_center_y + 224, average_center_x - 224:average_center_x + 224]
        cv2.imwrite('/home/pkrush/cents-test/' + str(coin_id).zfill(5) + str(count).zfill(2) + '.png', crop)
        count += 1

    return


def get_moving_center_x(frame, ratio, deskew_pixels, frame_name, frame_id):
    frame_width = int(1920 * ratio)
    frame_height = int(1080 * ratio)
    # print '3 In %s seconds' % (time.time() - start_time,)
    frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
    # print '4 In %s seconds' % (time.time() - start_time,)

    height_expansion_amount = 40
    blank_image = np.zeros((frame_height + height_expansion_amount, frame_width, 3), np.uint8)
    blank_image[height_expansion_amount / 2:frame_height + height_expansion_amount / 2, 0:frame_width] = frame
    frame = blank_image
    # frame = frame[460:,40:1040]
    deskewed = deskew(frame, deskew_pixels)
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 300, param1=45, param2=25, minRadius=52, maxRadius=58)
    if circles is None:
        cv2.imshow(frame_name, frame)
        return 0
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        center_x = i[0]
        center_y = i[1]
        crop_radius = i[2]
        cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 1)
        cv2.circle(frame, (center_x, center_y), crop_radius, (0, 0, 255), 1)
        # print circles
        cv2.imshow(frame_name, frame)
        cv2.imwrite('/home/pkrush/cents-circle-detect/' + str(frame_id).zfill(6) + frame_name + '.png', frame)
    return center_x * (1 / ratio)


def get_cameras():
    top_camera = None
    bottom_camera = None

    for camera_id in range(0, 3):
        cap = cv2.VideoCapture(camera_id)
        cap.set(3, 1920)
        cap.set(4, 1080)

        if cap.get(3) == 1920:
            if top_camera is None:
                top_camera = cap
            else:
                bottom_camera = cap

    top, bottom = read_from_cameras(top_camera, bottom_camera)
    if bottom[0, 0, 0] == bottom[0, 0, 1] == bottom[0, 0, 2]:
        temp_camera = top_camera
        top_camera = bottom_camera
        bottom_camera = temp_camera
    return top_camera, bottom_camera

top_camera, bottom_camera = get_cameras()

# files = glob.glob('/home/pkrush/cents-circle-detect/*')
# for f in files:
#     os.remove(f)
# files = glob.glob('/home/pkrush/cents-test/*')
# for f in files:
#     os.remove(f)

coin_id = 380
coin_is_starts = [0, 380]
# So this means 1, 378,381 are junk
ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200)
ser.write(str(102) + "\n")
cv.WaitKey(2)
ser.write(str(104) + "\n")
cv.WaitKey(2)
frame_count = 0
last_scan_frame_count = -100
found_coin = False

top_belt_on = True
bottom_belt_on = True

while (True):
    status = ''
    if top_belt_on and bottom_belt_on:
        # This might be overkill to keep turning them on:
        ser.write(str(102) + "\n")
        cv.WaitKey(1)
        ser.write(str(104) + "\n")
        cv.WaitKey(1)
    start_time = time.time()

    top, bottom = read_from_cameras(top_camera, bottom_camera)
    after_scan_frame_delay = 20
    if frame_count - last_scan_frame_count < after_scan_frame_delay:
        frame_count += 1
        continue

    if top_belt_on:
        center_x = get_moving_center_x(top, .1, 8, 'Top', frame_count)
        if center_x != 0:
            status += 'top' + ' ' + str(center_x) + '-'
            if top_belt_on and center_x < 1782:
                top_belt_on = False
                status += str(top_belt_on) + ' ' + str(bottom_belt_on) + '-'
                ser.write(str(105) + "\n")
                cv.WaitKey(1)
                status += 'Top belt off'

    if bottom_belt_on:
        center_x = get_moving_center_x(bottom, .11, -8, 'Bot', frame_count)
        if center_x != 0:
            status += 'bottom' + ' ' + str(center_x) + '-'
            if bottom_belt_on and center_x > 0:
                bottom_belt_on = False
                status += str(top_belt_on) + ' ' + str(bottom_belt_on) + '-'
                ser.write(str(103) + "\n")
                cv.WaitKey(1)
                status += 'Bottom belt off-'

    if top_belt_on == False and bottom_belt_on == False:
        # if first_top_scanned == True:
        status += 'Scanning ' + str(coin_id) + ' with the LED lights-'
        last_scan_frame_count = frame_count
        top_captures, bottom_captures = scan(top_camera, bottom_camera, ser)
        # t = threading.Thread(target=save, args=(top_captures, coin_id))
        # t.start()
        # t = threading.Thread(target=save, args=(bottom_captures, coin_id + 1))
        # t.start()
        save(top_captures, coin_id)
        save(bottom_captures, coin_id + 1)
        coin_id += 2
        #status += 'Cycle In %s seconds' % (time.time() - start_time,)
        start_time = time.time()
        status += 'Both belts on-'
        top_belt_on = True
        bottom_belt_on = True
        status += str(top_belt_on) + ' ' + str(bottom_belt_on) + '-'

    if status != '':
        print frame_count, status
    frame_count +=1

#ser.write(str(102) + "\n")
# cv.WaitKey(3500)
cv.WaitKey(35)
#ser.write(str(100) + "\n")
cv.WaitKey(100)
#ser.write(str(101) + "\n")

top_camera.release()
bottom_camera.release()
cv2.destroyAllWindows()
