#%matplotlib inline
import time

import numpy 			 as np
import numpy.matlib 	 as matlib
import matplotlib.pyplot as pyplot

from keras 				import utils
from keras.models 		import Sequential
from keras.layers 		import Dense, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping
from keras.datasets 	import mnist

# Import MNIST database from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

imgSizes = x[0].shape 		# Original image sizes
inputDim = x[0].size 		# Input dimension
m = x.shape[0]				# Dataset size
K = 10						# Number of classes = 10 for MNIST

# Unwrap input and labels
x = np.reshape(x, (m,inputDim))
y = utils.to_categorical(y, K)

# Shuffle dataset
index = np.random.permutation(m)
x = x[index]
y = y[index]

#debug
print("first x value sum :", x[0].sum())
#debug

# Train, test, validation split
# [------Train------/-Test-/-Val-]
trainSplit = 0.7
testSplit  = (1-trainSplit)/2
valSplit   = testSplit

trainIndex = np.floor(m*trainSplit).astype(int)
testIndex  = np.floor(m*testSplit).astype(int) + trainIndex

x_train = x[:trainIndex]
y_train = y[:trainIndex]

x_test  = x[trainIndex:testIndex]
y_test  = y[trainIndex:testIndex]

x_val   = x[testIndex:]
y_val   = y[testIndex:]

# Information about dimensions
print("")
print("Number of examples: ", m)
print("Input dimension: ", inputDim)

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

# Network architecture
neurons1 = 10

# Network hyperparameters
learningRate = 0.01
maxEpochs = 1000
batchSize = 256

#Input
model.add(Dense(units=neurons1, input_dim=inputDim))
model.add(Activation('tanh'))

#model.add(Dense(units=10))

# Output
model.add(Dense(units=10))
model.add(Activation('softmax'))


# Configure optimizer
sgd = SGD(lr=learningRate, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# Configure callbacks
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=6, verbose=1, mode='auto')


# Train Network
timerStart = time.time()

hist = model.fit(x_train, y_train, epochs=maxEpochs, batch_size=batchSize, callbacks=[earlyStop] ,validation_data=(x_val, y_val), verbose=0)
numEpochs = len(hist.history['acc'])

timerEnd = time.time()

eta = timerEnd-timerStart

# Test trained model
metrics = model.evaluate(x_test, y_test, batch_size=batchSize)
y_pred = model.predict(x_test, batch_size=batchSize)

# Information
print('\n')

#print(model.summary())

print('\n')
print("Epochs: ", numEpochs)
print("Elapsed time: ", eta)
print("Elapsed time per epoch: ", eta/numEpochs)
print("Loss: ", metrics[0])
print("Accuracy: ", metrics[1])

# Show predictions
print("----Predictions----")
print("\nPrediction :", np.argmax(y_pred[0]))
pyplot.imshow(np.reshape(x_test[0], (imgSizes)))