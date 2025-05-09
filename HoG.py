import os
import struct
import numpy as np
import gzip
import matplotlib.pyplot as plt
from skimage.feature import hog # you can install skimage by using  conda install scikit-image

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as impath:
        images = np.frombuffer(impath.read(), np.uint8, offset=16).reshape(len(labels),28,28)

    return images, labels


if __name__ == '__main__':

    # The shape of the processed training set X_train is 12000x28x28, where 12000 is the sample size. Replace the following address with your own dataset address.
    # The hog features of the test set can be obtained in the same way.
    X_train = np.load('/data2/orientliu/machine learning teaching assistant/homework3/X_train_sampled.npy')

    # Expand dimensions to 28x28x1 for an image x.
    # The following is an example of hog feature and visualization for an image. Replace the image with the training set or the test set (Set the hog function "visualize=False").
    print('image shape is', image.shape)

    # Get HoG feature h_x of an image
    arr, hog_image = hog(image, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=True)

    # Reshape HoG feature h_x into 784 dimensions
    hog_vector = hog_image.reshape(-1)
    print('hog_vector shape is', hog_vector.shape)