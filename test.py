#%matplotlib inline

import time

import numpy as numpy
import numpy.matlib as matlib
import matplotlib.pyplot as pyplot

from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import dataset

N = 2			# No of dimensions
L = 300			# Data size
K = 20			# No of classes
sigma = 0.25

dataSet = dataset.Dataset(N, L, K, sigma)

X, Y, T = dataSet.generate()

print(dataSet.X.shape)

Y_unwrap = dataSet.unwrap()
print("Y unwrap:", Y_unwrap.shape)
#dataSet.plotData()

## Configure Network
learningRate = 0.01

trainRatio = 0.6
testRatio = 0.2
valRatio = 0.2

batchSize = 128
maxEpochs = 1000

#numEpochs = 200

dataSize = dataSet.dataSize

model = Sequential()

# Split data in 3 sets
randIndex = numpy.random.permutation(numpy.arange(dataSize))

trainIndex = numpy.floor(dataSize*trainRatio).astype(int)
testIndex = numpy.floor(dataSize*(trainRatio+testRatio)).astype(int)

x_train = numpy.transpose(X[:,randIndex[:trainIndex]])
x_test  = numpy.transpose(X[:,randIndex[trainIndex:testIndex]])
x_val   = numpy.transpose(X[:,randIndex[testIndex:]])

y_train = numpy.transpose(Y_unwrap[:,randIndex[:trainIndex]])
y_test  = numpy.transpose(Y_unwrap[:,randIndex[trainIndex:testIndex]])
y_val   = numpy.transpose(Y_unwrap[:,randIndex[testIndex:]])

print("x train:, ", x_train.shape)
print("x test:, ", x_test.shape)
print("y train:, ", y_train.shape)
print("y test:, ", y_test.shape)

# Network architecture
model.add(Dense(units=10, input_dim=dataSet.dim))	# Input and one hidden layer
model.add(Activation('tanh'))

model.add(Dense(units=2))							# Output layer
model.add(Activation('softmax'))

# Configure optimizer
sgd = SGD(lr=learningRate, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])

# Configure callbacks
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=6, verbose=1, mode='auto')


# Train Network
timerStart = time.time()

hist = model.fit(x_train, y_train, epochs=maxEpochs, batch_size=batchSize, callbacks=[earlyStop] ,validation_data=(x_val, y_val), verbose=0)
numEpochs = len(hist.history['acc'])

timerEnd = time.time()

eta = timerEnd-timerStart

# Test trained model
metrics = model.evaluate(x_test, y_test, batch_size=2*batchSize)
y_pred = model.predict(x_test, batch_size=2*batchSize)

y_pred[:, 0] = numpy.where(y_pred[:, 0] >= 0.5, 1, 0)		# Negatives
y_pred[:, 1] = numpy.where(y_pred[:, 1] >= 0.5, 1, 0)	# Positives

y_pred = y_pred.astype(bool)

print("\ny_pred shape: ", y_pred.shape)

# Plot results

# posIndex = numpy.in1d(y_pred, 1)
# negIndex = numpy.in1d(y_pred, 0)
# print("\nposIndex shape: ", posIndex.shape)
# print("\nnegIndex shape: ", negIndex.shape)

pyplot.figure(figsize=(10,10))
pyplot.plot(x_test[y_pred[:, 1], 0], x_test[y_pred[:, 1], 1], 'r+')	# Plot positives
pyplot.plot(x_test[y_pred[:, 0], 0], x_test[y_pred[:, 0], 1], 'b+') # Plot negatives
pyplot.plot(T[0,:], T[1,:], 'ko', markersize=8, markerfacecolor='w', markeredgewidth=2)

targetClasses = dataSet.targetClasses

pyplot.plot(T[0,targetClasses], T[1,targetClasses], 'kx', markersize=13, markeredgewidth=2)

# Information
print('\n')

#print(model.summary())

print('\n')
print("Epochs: ", numEpochs)
print("Elapsed time: ", eta)
print("Elapsed time per epoch: ", eta/numEpochs)
print("Loss: ", metrics[0])
print("Accuracy: ", metrics[1])