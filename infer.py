import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

# Make sure that caffe is on the python path:
# sys.path.append('~/caffe/python') using the ~ does not work, for some reason???
sys.path.append('/home/pkrush/caffe/python')
import caffe


def get_classifier(model_name, crop_size):
    model_dir = model_name + '/'
    image_dir = 'test-images/'
    MODEL_FILE = model_dir + 'deploy.prototxt'
    PRETRAINED = model_dir + 'snapshot.caffemodel'
    meanFile = model_dir + 'mean.binaryproto'

    # Open mean.binaryproto file
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(meanFile, 'rb').read()
    blob.ParseFromString(data)
    mean_arr = np.array(caffe.io.blobproto_to_array(blob)).reshape(1, crop_size, crop_size)
    print mean_arr.shape

    net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(crop_size, crop_size), mean=mean_arr, raw_scale=255)
    return net;


def get_labels(model_name):
    labels_file = model_name + '/labels.txt'
    labels = [line.rstrip('\n') for line in open(labels_file)]
    return labels;


def get_caffe_image(crop, crop_size):
    # this is how you get the image from file:
    # coinImage = [caffe.io.load_image("some file", color=False)]

    caffe_image = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    caffe_image = caffe_image.astype(np.float32) / 255
    caffe_image = np.array(caffe_image).reshape(crop_size, crop_size, 1)
    # Caffe wants a list so []:
    return [caffe_image];


def rotate(img, angle):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cv2.warpAffine(img, M, (cols, rows),img, cv2.INTER_CUBIC)
    return img;

def crop_for_date(src):
    dst = src[307:307 + 64, 250:250 + 64]
    dst = src[250:250 + 64, 307:307 + 64]
    dst = cv2.resize(dst, (28, 28), interpolation=cv2.INTER_AREA)
    return dst;


cap = cv2.VideoCapture(1)
copper60 = get_classifier("copper60", 60)
heads_with_rotation64 = get_classifier("heads-with-rotation64", 64)
dates_over_50 = get_classifier("dates-over-50", 28)
count = 0
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[27:433, 117:523]
    cv2.imshow('frame', gray)

    copper60_score = copper60.predict(get_caffe_image(gray, 60), oversample=False)
    #print copper60_score
    heads_with_rotation64_score = heads_with_rotation64.predict(get_caffe_image(gray, 64), oversample=False)
    #print heads_with_rotation64_score
    count = count + 1
    #print count
    max_value = np.amax(heads_with_rotation64_score)
    angle = np.argmax(heads_with_rotation64_score)
    rotated = rotate(gray,360-angle)
    cv2.imshow('rotated', rotated)
    #print max_value,angle
    dateCrop = crop_for_date(rotated)
    cv2.imshow('dateCrop', dateCrop)

    dates_over_50_score = dates_over_50.predict(get_caffe_image(dateCrop, 28), oversample=False)
    #print dates_over_50_score
    date_labels = get_labels("dates-over-50")
    predicted_date = date_labels[np.argmax(dates_over_50_score)]
    #print predicted_date
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rotated, predicted_date, (300,240),font, 1,(0,0,0),2)

    cv2.imshow('rotated', rotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
