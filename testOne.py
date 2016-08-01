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
MODEL_FILE = model_dir + 'deploy.prototxt'
PRETRAINED = model_dir + 'snapshot.caffemodel'
IMAGE_FILE = 'temp.png'

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(28, 28), raw_scale=255)
score = net.predict([caffe.io.load_image(IMAGE_FILE, color=False)], oversample=False)
print score
