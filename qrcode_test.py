#This is just used to see the webcam output

import time
import cv2
import zbar
import Image
import numpy as np
import pyqrcode
import glob
import random
import pickle

dir = "/home/pkrush/find-parts-faster/"
output_width = 960
output_height = 540
background_width = 1280
background_height = 720
padding = 20
qr_scale = 8
qr_unscaled_size = 21  # version 1
qr_size = qr_unscaled_size * qr_scale
x_qr_interval = background_width - (padding * 2) - qr_size
y_qr_interval = background_height - (padding * 2) - qr_size
yes_points = []
no_points = []


def get_qr_code(data,scale):
    qr = pyqrcode.create(data, error='H', version=1, mode='binary')
    qr_code = np.array(qr.code, dtype=np.uint8)
    qr_code[qr_code == 0] = 255
    qr_code[qr_code == 1] = 0
    qr_code_size = qr_code.shape[0] * scale
    qr_code = cv2.resize(qr_code, (qr_code_size, qr_code_size), interpolation=cv2.INTER_AREA)
    return qr_code

def mark(event, x, y, flags, param):
    global yes_points
    global no_points

    if event == cv2.EVENT_LBUTTONDOWN:
        yes_points.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        no_points.append((x, y))

def get_points_inside_border(points,border):
    new_points = []
    for x,y in points:
        if x < border:
            continue
        if x > output_width - border:
            continue
        if y < border:
            continue
        if y > output_height - border:
            continue
        new_points.append((x,y))
    return new_points

def label_warped():
    cv2.namedWindow("Warped")
    cv2.setMouseCallback("Warped", mark)
    global yes_points
    global no_points

    yes_points = pickle.load(open(dir + 'yes_points.pickle', "rb"))
    no_points = pickle.load(open(dir + 'no_points.pickle', "rb"))

    warped_filenames = []
    for filename in glob.iglob(dir + 'warped/' + '*.png'):
        #crops.append([random.random(), filename])
        warped_filenames.append(filename)
    warped_filenames.sort()
    loop = True
    while loop:
        for filename in warped_filenames:
            yes_points = get_points_inside_border(yes_points, 28)
            no_points = get_points_inside_border(no_points, 28)

            warped = cv2.imread(filename)
            for marked_point in yes_points:
                cv2.circle(warped,marked_point, 28, (255, 255,255), 1)
            for marked_point in no_points:
                cv2.circle(warped,marked_point, 28, (0,0,0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(warped, 'Yes:' + str(len(yes_points)) + ' No:' + str(len(no_points)) , (4, 20), font, .7, (0, 255, 0), 2)
            cv2.imshow("Warped", warped)
            if cv2.waitKey(4) & 0xFF == ord('q'):
                loop = False
                break
    pickle.dump(yes_points, open(dir + 'yes_points.pickle', "wb"))
    pickle.dump(no_points, open(dir + 'no_points.pickle', "wb"))


def save_crops(warped, points,filename_start):
    for x, y in points:
        crop0 = warped[y - 28:y + 28, x - 28:x + 28]
        crop = crop0.copy()
        angle = random.random() * 360
        m = cv2.getRotationMatrix2D((28, 28), angle, 1)
        cv2.warpAffine(crop, m, (56, 56), crop, cv2.INTER_CUBIC)
        crop_filename = filename_start + 'id' + str(x).zfill(4) + 'x' + str(y).zfill(4) + 'y' + '.png'
        cv2.imwrite(crop_filename, crop)


def fill_train_dir():
    global yes_points
    global no_points

    yes_points = pickle.load(open(dir + 'yes_points.pickle', "rb"))
    no_points = pickle.load(open(dir + 'no_points.pickle', "rb"))

    warped_filenames = []
    for filename in glob.iglob(dir + 'warped/' + '*.png'):
        #crops.append([random.random(), filename])
        warped_filenames.append(filename)
        warped_filenames.sort()

    for filename in warped_filenames:
        warped = cv2.imread(filename)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_id = str(filename).replace('.png','')
        warped_id = warped_id[(len(warped_id)-5):] #last 5 is ID

        filename_start = dir + 'train/yes/'+  warped_id
        save_crops(warped, yes_points,filename_start)
        filename_start = dir + 'train/no/' + warped_id
        save_crops(warped, no_points, filename_start)

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
    cap = cv2.VideoCapture(dir + "1.mp4")
    #cap.set(3,1920)
    #cap.set(4,1080)

    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')

    for count in range(0,400000):
        # Capture frame-by-frame
        start_time = time.time()
        ret, frame = cap.read()

        if frame == None:
            break
        #cv2.imwrite(str(x) + '.png', frame)

        output = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dstCn=0)
        print '0 in %s seconds' % (time.time() - start_time,)

        pil = Image.fromarray(gray)
        width, height = pil.size
        raw = pil.tostring()
        print '1 in %s seconds' % (time.time() - start_time,)

        # create a reader
        image = zbar.Image(width, height, 'Y800', raw)
        print '2 in %s seconds' % (time.time() - start_time,)
        scanner.scan(image)
        print '3 in %s seconds' % (time.time() - start_time,)

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

                max_dst[0] = [offset_x, offset_y]
                max_dst[1] = [offset_x, y_qr_interval + qr_size]
                max_dst[2] = [x_qr_interval + qr_size, y_qr_interval + qr_size]
                max_dst[3] = [x_qr_interval + qr_size, offset_y]

                M = cv2.getPerspectiveTransform(max_src, max_dst)
                warped = cv2.warpPerspective(output, M, (background_width-(padding*2),background_height-(padding*2)))
                warped = cv2.resize(warped, (output_width, output_height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dir + 'warped/' + str(count).zfill(5) + '.png', warped)
                cv2.imshow("warped", warped)

        output = cv2.resize(output, (output_width, output_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Camera", output)
        print '4 in %s seconds' % (time.time() - start_time,)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #print 'In %s seconds' % (time.time() - start_time,)

    cap.release()
    cv2.destroyAllWindows()

#display_background()
#process_video()
#label_warped()
fill_train_dir()