import numpy
import numpy.matlib as matlib
import matplotlib.pyplot as pyplot

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers


## Generate random vectors with gaussian clouds
print('\n')
print('-- Generate random clusters --')

N = 2			# No of dimensions
K = 20			# No of classes
L = 100			# Data size
dataSize = K*L 	# Dataset size

sigma = 0.25	# Gaussian variance, sigma²

T = numpy.random.randn(N, K) # Cluster centers
print('T shape: ', T.shape)

X = matlib.repmat(T, 1, L) + sigma*numpy.random.randn(N, K*L)
print('X shape: ', X.shape)

Y = matlib.repmat(range(K),1,L)
print('Y shape: ', Y.shape)

## Plot the Gaussian clusters
print('\n')
print('-- Unwrap targets --')

targetClasses = numpy.random.permutation(numpy.arange(K))[:numpy.floor(K/2).astype(int)]	# K/2 array of target classes

targetIndex = numpy.in1d(Y,targetClasses) # Indexes of I/O pairs belonging to target classes
targetLabels = X[:,targetIndex]			  # Input elements belonging to target classes

Y_unwrap = numpy.zeros(Y.shape)
Y_unwrap[:,targetIndex] = 1

print('Target classes shape: ', targetClasses.shape)
print('Target labels shape: ', targetLabels.shape)
print('Y_unwrap shape: ', Y_unwrap.shape)

print('\n')
print('-- Run NN --')

pyplot.figure(figsize=(10,10))
pyplot.plot(X[0,:], X[1,:], 'b+')
pyplot.plot(T[0,:], T[1,:], 'ko', markersize=8, markerfacecolor='w', markeredgewidth=2)
pyplot.title('Random Clusters')

pyplot.plot(targetLabels[0,:], targetLabels[1,:], 'r+')
pyplot.plot(T[0,targetClasses], T[1,targetClasses], 'kx', markersize=13, markeredgewidth=2)

## Configure Network
learningRate = 0.01

trainRatio = 0.6
testRatio = 0.2
valRatio = 0.2

model = Sequential()

# Split data in 3 sets
#X = numpy.transpose(X)
#Y = numpy.transpose(Y)

dataIndex = numpy.random.permutation(numpy.arange(dataSize))

x_train = numpy.transpose(X[:,dataIndex[:numpy.floor(dataSize*trainRatio).astype(int)]])
x_test = numpy.transpose(X[:,dataIndex[numpy.floor(dataSize*trainRatio).astype(int):numpy.floor(dataSize*(trainRatio+testRatio)).astype(int)]])
x_val = numpy.transpose(X[:,dataIndex[numpy.floor(dataSize*(trainRatio+testRatio)).astype(int):]])

y_train = numpy.transpose(X[:,dataIndex[:numpy.floor(dataSize*trainRatio).astype(int)]])
y_test = numpy.transpose(X[:,dataIndex[numpy.floor(dataSize*trainRatio).astype(int):numpy.floor(dataSize*(trainRatio+testRatio)).astype(int)]])
y_val = numpy.transpose(X[:,dataIndex[numpy.floor(dataSize*(trainRatio+testRatio)).astype(int):]])

# Input and one hidden layer
model.add(Dense(units=10, input_dim=2))
model.add(Activation('sigmoid'))

# Output layer
model.add(Dense(units=2))
model.add(Activation('softmax'))

# Configure optimizer
sgd = optimizers.SGD(lr=learningRate, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

print("Model summary:\n")
model.summary()

model.fit(x_train, y_train, epochs=5, batch_size=10)

metrics = model.evaluate(x_test, y_test, batch_size=20)

print("Metrics: ", metrics)