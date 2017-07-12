import numpy as numpy
import numpy.matlib as matlib
import matplotlib.pyplot as pyplot

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


import dataset

N = 2			# No of dimensions
L = 100			# Data size
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

dataSize = dataSet.dataSize

model = Sequential()

# Split data in 3 sets
randIndex = numpy.random.permutation(numpy.arange(dataSize))

trainIndex = numpy.floor(dataSize*trainRatio).astype(int)
testIndex = numpy.floor(dataSize*(trainRatio+testRatio)).astype(int)

x_train = numpy.transpose(X[:,randIndex[:trainIndex]])
x_test  = numpy.transpose(X[:,randIndex[trainIndex:testIndex]])
x_val   = numpy.transpose(X[:,randIndex[testIndex:]])

# x_train = X[:,randIndex[:trainIndex]]
# x_test  = X[:,randIndex[trainIndex:testIndex]]
# x_val   = X[:,randIndex[testIndex:]]

y_train = numpy.transpose(Y[:,randIndex[:trainIndex]])
y_test  = numpy.transpose(Y[:,randIndex[trainIndex:testIndex]])
y_val   = numpy.transpose(Y[:,randIndex[testIndex:]])


print("x train:, ", x_train.shape)
print("x test:, ", x_test.shape)
print("y train:, ", y_train.shape)
print("y test:, ", y_test.shape)

# Input and one hidden layer
model.add(Dense(units=10, input_dim=dataSet.dim))
model.add(Activation('sigmoid'))

# Output layer
model.add(Dense(units=2))
model.add(Activation('softmax'))

# Configure optimizer
sgd = SGD(lr=learningRate, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=10)

metrics = model.evaluate(x_test, y_test, batch_size=20)

print("Metrics: ", metrics)