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

print("first value sum :", x_train[0].sum())
print("y_train: ", y_train[0:10])

testSize = x_test.shape[0] # temporary

x_val = x_test[0:np.floor(testSize/2).astype(int)]
y_val = y_test[0:np.floor(testSize/2).astype(int)]

x_test = x_test[np.floor(testSize/2).astype(int):]
y_test = y_test[np.floor(testSize/2).astype(int):]

m = x_train.shape[0]		# Training data size
inputDim = x_train[0].size 	# Input dimension
testSize = x_test.shape[0]	# Test and validation data sizes
valSize  = x_val.shape[0]
K = 10						# Number of classes

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

x_trainUnwrap = np.reshape(x_train, (m,inputDim))
#y_trainUnwrap = np.ravel(y_train)
x_testUnwrap  = np.reshape(x_test, (testSize, inputDim))
#y_testUnwrap  = np.ravel(y_test)
x_valUnwrap  = np.reshape(x_val, (valSize, inputDim))

print("")
print("Shape x_trainUnwrap: ", x_trainUnwrap.shape)
print("Shape x_testUnwrap: ", x_testUnwrap.shape)

# Unwrap labels
y_trainUnwrap = utils.to_categorical(y_train, K)
y_testUnwrap = utils.to_categorical(y_test, K)
y_valUnwrap = utils.to_categorical(y_val, K)


## Network
model = Sequential()

#Input
model.add(Dense(units=50, input_dim=inputDim))
model.add(Activation('tanh'))

#model.add(Dense(units=10))

# Output
model.add(Dense(units=10))
model.add(Activation('softmax'))

# Network hyperparameters
learningRate = 0.01
maxEpochs = 1000
batchSize = 256

# Configure optimizer
sgd = SGD(lr=learningRate, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# Configure callbacks
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=6, verbose=1, mode='auto')


# Train Network
timerStart = time.time()

hist = model.fit(x_trainUnwrap, y_trainUnwrap, epochs=maxEpochs, batch_size=batchSize, callbacks=[earlyStop] ,validation_data=(x_valUnwrap, y_valUnwrap), verbose=0)
numEpochs = len(hist.history['acc'])

timerEnd = time.time()

eta = timerEnd-timerStart

# Test trained model
metrics = model.evaluate(x_testUnwrap, y_testUnwrap, batch_size=batchSize)
y_pred = model.predict(x_testUnwrap, batch_size=batchSize)

# Information
print('\n')

#print(model.summary())

print('\n')
print("Epochs: ", numEpochs)
print("Elapsed time: ", eta)
print("Elapsed time per epoch: ", eta/numEpochs)
print("Loss: ", metrics[0])
print("Accuracy: ", metrics[1])