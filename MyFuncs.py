import numpy as np
from skimage.feature import hog

def load_data():
    X_train = np.load('X_train_sampled.npy')
    X_test = np.load('X_test_sampled.npy')
    y_train = np.load('y_train_sampled.npy')
    y_test = np.load('y_test_sampled.npy')

    x_train_hogs = []
    for x_train_hog in X_train:
        arr, hog_image = hog(x_train_hog, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=True)
        x_train_hogs.append(arr)
    x_test_hogs = []
    for x_test_hog in X_test:
        arr, hog_image = hog(x_test_hog, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=True)
        x_test_hogs.append(arr)

    return x_train_hogs, y_train, x_test_hogs, y_test