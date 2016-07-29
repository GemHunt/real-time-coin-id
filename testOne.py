import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.append('~/caffe/python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'mnist/lenet.prototxt'
PRETRAINED = 'mnist/lenet_iter_10000.caffemodel'
IMAGE_FILE = 'mnist/9.png'

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(28, 28), raw_scale=255)
score = net.predict([caffe.io.load_image(IMAGE_FILE, color=False)], oversample=False)
print score
