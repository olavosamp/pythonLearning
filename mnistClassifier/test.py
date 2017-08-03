# %matplotlib inline

import numpy 			 as np
import numpy.matlib 	 as matlib
import matplotlib.pyplot as pyplot

from keras 				import utils
from keras.models 		import Sequential
from keras.layers 		import Dense, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping
from keras.datasets 	import mnist

# A = np.array([[1], [1,2,3,4,5], [1,2,3,4,5]])

# B = np.array([[2], [1,2,3,4,5], [1,2,3,4,5]])

# print("")
# print(A)

# print(B)
# C = np.concatenate((A, B))
# print(C)

# print("")
# print("A shape: ", A.shape)
# print("B shape: ", B.shape)
# print("C shape: ", C.shape)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

pyplot.imshow(x_train[0])