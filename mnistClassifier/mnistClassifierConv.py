#%matplotlib inline
import time

import numpy 			 as np
import numpy.matlib 	 as matlib
import matplotlib.pyplot as pyplot

from keras 				import utils
from keras.models 		import Sequential
from keras.layers 		import Dense, Activation, Conv2D, ZeroPadding2D, MaxPooling2D
#from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping
from keras.datasets 	import mnist

import dataset as dataset

weightsPath = ".\\weights\\" + "leNet5-11"

# # Import MNIST database from Keras
x, y, imgSize = dataset.loadMnist()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x = np.concatenate((x_train, x_test))
# y = np.concatenate((y_train, y_test))

#imgSizes = np.squeeze(x[0].shape) 	# Original image sizes
K = 10								# Number of classes = 10 for MNIST
m = x.shape[0]						# Dataset size

# # Convert input to rgb and unwrap labels
# x = x[:,:,:,np.newaxis]
# #x = np.repeat(x, 3, 3)

# y = utils.to_categorical(y, K)
# y = np.reshape(y, (m, 1, 1, K))

# # Using just a few examples
# # x = x[:1000]
# # y = y[:1000]
# # m = x.shape[0]				# Dataset size

# # Shuffle dataset
# index = np.random.permutation(m)
# x = x[index]
# y = y[index]

#debug
print("first x value sum :", x[0].sum())
#debug

# Train, test, validation split
# [------Train------/-Test-/-Val-]
trainSplit = 0.6

x_train, y_train, x_test, y_test, x_val, y_val = dataset.dataSplit(x, y, trainSplit)
# testSplit  = (1-trainSplit)/2
# valSplit   = testSplit

# trainIndex = np.floor(m*trainSplit).astype(int)
# testIndex  = np.floor(m*testSplit).astype(int) + trainIndex

# x_train = x[:trainIndex]
# y_train = y[:trainIndex]

# x_test  = x[trainIndex:testIndex]
# y_test  = y[trainIndex:testIndex]

# x_val   = x[testIndex:]
# y_val   = y[testIndex:]

# Information about dimensions
print("")
print("Number of examples: ", m)
#print("Input shape: ", inputShape)

print("")
print("Shape x_train: ", x_train.shape)
print("Shape y_train: ", y_train.shape)

print("Shape x_test: ", x_test.shape)
print("Shape y_test: ", y_test.shape)

print("Shape x_val: ", x_val.shape)
print("Shape y_val: ", y_val.shape)
print("")

## Network
model = Sequential()

# Architecture
	# LeNet-5
	# in-C1-S2-C3-S4-C5-F6-out

# Hyperparameters
learningRate = 0.01
maxEpochs = 300
batchSize = 1024

inputShape = x_train[0].shape 	# Input dimension
print("Input shape: ", inputShape)

#Input
model.add(ZeroPadding2D(padding=(2,2), input_shape=inputShape))

# C1
model.add(Conv2D(filters=6, kernel_size=5))
model.add(Activation('relu'))

# S2
model.add(MaxPooling2D(pool_size=2, strides=2))

# C3
model.add(Conv2D(filters=16, kernel_size=5))
model.add(Activation('relu'))

# S4
model.add(MaxPooling2D(pool_size=2, strides=2))

# C5
model.add(Conv2D(filters=120, kernel_size=5))
model.add(Activation('relu'))

# F6
model.add(Dense(units=84))
model.add(Activation('relu'))

# Output
model.add(Dense(units=10))
model.add(Activation('softmax'))

# Configure optimizer
# sgd = SGD(lr=learningRate, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Configure callbacks
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=6, verbose=1, mode='auto')

print(model.summary())

# Train Network
timerStart = time.time()

hist = model.fit(x_train, y_train, epochs=maxEpochs, batch_size=batchSize, callbacks=[earlyStop] ,validation_data=(x_val, y_val), verbose=1)
numEpochs = len(hist.history['acc'])

timerEnd = time.time()

eta = timerEnd-timerStart

# Test trained model
metrics = model.evaluate(x_test, y_test, batch_size=batchSize)
y_pred = model.predict(x_test, batch_size=batchSize)

# Save weights
model.save_weights(weightsPath)

# Information
print('\n')

print('\n')
print("Epochs: ", numEpochs)
print("Elapsed time: ", eta)
print("Elapsed time per epoch: ", eta/numEpochs)
print("Loss: ", metrics[0])
print("Accuracy: ", metrics[1])

# Show predictions
print("----Predictions----")
print("\nPrediction :", np.argmax(y_pred[0]))
pyplot.imshow(np.reshape(x_test[0], (imgSize)))