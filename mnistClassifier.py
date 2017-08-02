#%matplotlib inline

import time

import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as pyplot

from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from keras.datasets import mnist

# Import MNIST database from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("\nShape x_train: ", x_train.shape)
print("Shape x_test: ", x_test.shape)

print("Shape y_train: ", y_train.shape)
print("Shape y_test: ", y_test.shape)

test = np.array([[1, 2, 3], [4, 4, 4], [5, 5, 5]])

print("\n", test)

print("\nTest ravel: ", np.ravel(test))

inputs = Input(shape=test.size)