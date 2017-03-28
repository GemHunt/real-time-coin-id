import numpy as np
import os
import serial
import time
import sys
import cv2
import cv2.cv as cv
import cPickle as pickle


def get_filename(coin_id, image_id):
    dir = '/home/pkrush/cents-test/' + str(coin_id / 100) + '/'
    filename = dir + str(coin_id).zfill(5) + str(image_id).zfill(2) + '.png'
    return filename

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
    crop_radius = 224
    border_expansion = 30
    center_list = []
    resized = []
    start_time = time.time()

    for frame in captures:
        if coin_id % 2 == 0:
            ratio = .41
        else:
            ratio = .46

        frame_width = int(1920 * ratio)
        frame_height = int(1080 * ratio)
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        blank_image = np.zeros((frame_height + border_expansion * 2, frame_width + border_expansion * 2, 3), np.uint8)
        blank_image[border_expansion:frame_height + border_expansion,
        border_expansion:frame_width + border_expansion] = frame
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
            coin_radius = i[2]
            cv2.circle(gray, (center_x, center_y), 2, (0, 0, 255), 1)
            cv2.circle(gray, (center_x, center_y), coin_radius, (0, 0, 255), 1)
            center_list.append([center_x, center_y, coin_radius])

    total_center_x = 0
    total_center_y = 0
    total_radius = 0

    # print '1 In %s seconds' % (time.time() - start_time,)

    for center_x, center_y, coin_radius in center_list:
        # print center_x, center_y, coin_radius
        total_center_x += center_x
        total_center_y += center_y
        total_radius += coin_radius
    #print '2 In %s seconds' % (time.time() - start_time,)

    if len(center_list) == 0:
        return False
        # raise ValueError(str(coin_id) + 'had no detected circles')
    #print '3 In %s seconds' % (time.time() - start_time,)

    average_center_x = float(total_center_x) / len(center_list)
    average_center_y = float(total_center_y) / len(center_list)
    average_radius = float(total_radius) / len(center_list)

    resized_height,resized_width,channels = frame.shape
    crop_top = average_center_y - crop_radius
    crop_bottom = average_center_y + crop_radius
    crop_left = average_center_x - crop_radius
    crop_right = average_center_x + crop_radius
    bad_crop =  ' is Bad. X&Y:' + str(average_center_x) + "," + str(average_center_y) + ' Frame Width:' + str(resized_width) + ' Frame Height:' + str(resized_height)

    if crop_left < 0:
        print str(crop_left) + ' crop_left' + bad_crop + '\n\n\n'
        #return False

    if crop_right > resized_width:
        print str(crop_right) + ' crop_right' +  bad_crop + '\n\n\n'
        #return False

    if crop_top < 0:
        print str(crop_top) + ' crop_top' + bad_crop + '\n\n\n'
        #return False

    if crop_bottom > resized_height:
        print str(crop_bottom) + ' crop_bottom' + bad_crop + '\n\n\n'
        #return False

    # dir = '/media/pkrush/Seagate Backup Plus Drive/cents_2/' + str(coin_id/100) + '/'
    dir = '/home/pkrush/cents-test/' + str(coin_id / 100) + '/'

    if not os.path.exists(dir):
        os.mkdir(dir)
    #print '5 In %s seconds' % (time.time() - start_time,)

    for frame in resized:
        crop = frame[crop_top:crop_bottom, crop_left:crop_right]
        cv2.imwrite(dir + str(coin_id).zfill(5) + str(count).zfill(2) + '.png', crop)
        count += 1
    #print '6 In %s seconds' % (time.time() - start_time,)

    return True

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

    for camera_id in range(0, 4):
        cap = cv2.VideoCapture(camera_id)
        cap.set(3, 1920)
        cap.set(4, 1080)

        if cap.get(3) == 1920:
            if top_camera is None:
                top_camera = cap
            else:
                bottom_camera = cap

    top, bottom = read_from_cameras(top_camera, bottom_camera)
    if bottom[170, 170, 0] == bottom[170, 170, 1] == bottom[170, 170, 2]:
        temp_camera = top_camera
        top_camera = bottom_camera
        bottom_camera = temp_camera
    return top_camera, bottom_camera

#this is a one time function as the init scanning had issues.
#237 sets of 2 were bad 2500 were good. I have 5000 good sets of 57 images for 2500 coins.
def save_good_coin_ids():
    good_coin_ids = set()
    bad_coin_ids = set()
    #for coin_id in range(0, 5458, 2):
    for coin_id in range(0, 5458, 2):
        good_coin_ids.add(coin_id)
        for side in [0, 3]:
            for image_id in range(0, 56):
                filename = get_filename(coin_id + side, image_id)
                if not os.path.isfile(filename):
                    bad_coin_ids.add(coin_id)
                    continue
                if os.path.getsize(filename) == 0:
                    bad_coin_ids.add(coin_id)
                    continue
            test_image = cv2.imread(filename)
            if test_image is None:
                bad_coin_ids.add(coin_id)
                continue

            width, height, channels = test_image.shape
            if not width == height == 448:
                bad_coin_ids.add(coin_id)
                continue

    good_coin_ids = good_coin_ids - bad_coin_ids
    for start_id in coin_id_starts:
        if start_id != 0:
            #-2 is bad: Why;
            #-2 bad the for top coin_id is good,
            #-1 good the bottom of -4 good,
            #0 good top coin_id is good,
            #1 bad bottom will never be read as it's the back of -2
            #2 good top is new the back of 0
            #3 good bottom is the back of #0
            bad_coin_ids.add(start_id - 2)

    print len(bad_coin_ids)
    print len(good_coin_ids)
    good_coin_ids.difference(bad_coin_ids)
    home_dir = '/home/pkrush/cent-models/'
    data_dir = home_dir + 'metadata/'
    back_sides = set()
    for coin_id in good_coin_ids:
        back_sides.add(coin_id + 3)
    good_coin_ids = good_coin_ids.union(back_sides)
    print len(good_coin_ids)
    pickle.dump(good_coin_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))
    pickle.dump(good_coin_ids, open(data_dir + 'test_image_ids.pickle', "wb"))


coin_id_starts = [0, 380, 1152, 1972, 2674, 2780, 2846, 2946, 3330, 5448]
def get_start_coin_id():
    return coin_id_starts[len(coin_id_starts) - 1]


save_good_coin_ids()
sys.exit()

coin_id = get_start_coin_id()
top_camera, bottom_camera = get_cameras()

# files = glob.glob('/home/pkrush/cents-circle-detect/*')
# for f in files:
#     os.remove(f)
# files = glob.glob('/home/pkrush/cents-test/*')
# for f in files:
#    os.remove(f)

start_time = time.time()

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

    top, bottom = read_from_cameras(top_camera, bottom_camera)
    after_scan_frame_delay = 30
    if frame_count - last_scan_frame_count < after_scan_frame_delay:
        frame_count += 1
        continue

    if top_belt_on:
        center_x = get_moving_center_x(top, .1, 8, 'Top', frame_count)
        if center_x != 0:
            status += 'top' + ' ' + str(center_x) + '-'
            if top_belt_on and center_x < 1691:
                top_belt_on = False
                status += str(top_belt_on) + ' ' + str(bottom_belt_on) + '-'
                ser.write(str(105) + "\n")
                cv.WaitKey(1)
                ser.write(str(106) + "\n")
                cv.WaitKey(10)
                ser.write(str(107) + "\n")
                cv.WaitKey(1)
                status += 'Top belt off, reset hopper'

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
        # print 'pre save In %s seconds' % (time.time() - start_time,)
        top_save = save(top_captures, coin_id)
        bottom_save = save(bottom_captures, coin_id + 1)
        # print 'save In %s seconds' % (time.time() - start_time,)

        if top_save and bottom_save:
            coin_id += 2

        status += 'Cycle In %s seconds' % (time.time() - start_time,)
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
