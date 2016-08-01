import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
#caffe_root = '/home/pkrush/'  # this file is expected to be in {caffe_root}/examples
import sys
#sys.path.insert(0, caffe_root + 'python')
sys.path.append('/home/pkrush/caffe/python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
model_dir = 'copper60/'
image_dir = 'test-images/'
MODEL_FILE = model_dir + 'deploy.prototxt'
PRETRAINED = model_dir + 'snapshot.caffemodel'
meanFile = model_dir + 'mean.binaryproto'
IMAGE_FILE = image_dir +  'temp.png' 


# Open mean.binaryproto file
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(meanFile , 'rb').read()
blob.ParseFromString(data)
mean_arr = np.array(caffe.io.blobproto_to_array(blob)).reshape(1,60,60)
print mean_arr.shape

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(60, 60), mean = mean_arr, raw_scale=255)

coinImage = [caffe.io.load_image(IMAGE_FILE, color=False)]

for sec in range(1,10):
	for num in range(1,30):
		score = net.predict(coinImage, oversample=False)
	print score

