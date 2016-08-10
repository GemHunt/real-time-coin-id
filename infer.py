import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

# Make sure that caffe is on the python path:
# sys.path.append('~/caffe/python') using the ~ does not work, for some reason???
sys.path.append('/home/pkrush/caffe/python')
import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
model_dir = 'copper60/'
image_dir = 'test-images/'
MODEL_FILE = model_dir + 'deploy.prototxt'
PRETRAINED = model_dir + 'snapshot.caffemodel'
meanFile = model_dir + 'mean.binaryproto'
IMAGE_FILE = image_dir + 'temp.png'

# Open mean.binaryproto file
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(meanFile, 'rb').read()
blob.ParseFromString(data)
mean_arr = np.array(caffe.io.blobproto_to_array(blob)).reshape(1, 60, 60)
print mean_arr.shape

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(60, 60), mean=mean_arr, raw_scale=255)

cap = cv2.VideoCapture(1)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[40:470, 110:540]
    gray = cv2.resize(gray, (60, 60), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255

    # Display the resulting frame
    cv2.imshow('frame', gray)

    coinImage = [caffe.io.load_image(IMAGE_FILE, color=False)]

    print "shap1"
    print gray.shape
    gray = np.array(gray).reshape(60, 60, 1)
    # print "gray"
    # print gray.shape
    coinImage2 = [gray]
    score = net.predict(coinImage2, oversample=False)
    print score

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
