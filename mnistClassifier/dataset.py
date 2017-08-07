import numpy 			 as np

from keras 				import utils
from keras.datasets 	import mnist

def loadMnist():
		# Import MNIST database from Keras
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x = np.concatenate((x_train, x_test))
	y = np.concatenate((y_train, y_test))

	imgSize = np.squeeze(x[0].shape) 	# Original image sizes
	K = 10						# Number of classes = 10 for MNIST
	m = x.shape[0]				# Dataset size

	# Convert input to rgb and unwrap labels
	x = x[:,:,:,np.newaxis]
	#x = np.repeat(x, 3, 3)

	y = utils.to_categorical(y, K)
	y = np.reshape(y, (m, 1, 1, K))

	# Using just a few examples
	# x = x[:1000]
	# y = y[:1000]
	# m = x.shape[0]				# Dataset size

	# Shuffle dataset
	index = np.random.permutation(m)
	x = x[index]
	y = y[index]

	return x, y, imgSize

def dataSplit(x, y, trainSplit, testSplit=0, valSplit=0):
	m = x.shape[0]				# Dataset size

	if testSplit == 0:
		testSplit  = (1-trainSplit)/2

	if valSplit == 0:
		valSplit   = testSplit

	trainIndex = np.floor(m*trainSplit).astype(int)
	testIndex  = np.floor(m*testSplit).astype(int) + trainIndex

	x_train = x[:trainIndex]
	y_train = y[:trainIndex]

	x_test  = x[trainIndex:testIndex]
	y_test  = y[trainIndex:testIndex]

	x_val   = x[testIndex:]
	y_val   = y[testIndex:]

	return x_train, y_train, x_test, y_test, x_val, y_val