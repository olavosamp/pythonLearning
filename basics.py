import numpy
import numpy.matlib as matlib
import matplotlib.pyplot as pyplot

from keras.models import Sequential

## Algebra basics
# Create a vector and a matrix

A = numpy.array([1, 2, 3])

print('\n')
print('-- Create Vector and Matrix --')

print('Dim: ', A.ndim, '\n')
print('Shape: ', A.shape, '\n')
print('Element (1,2): ', A[2], '\n')

B = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

print(B)


# Matrix sum and multiplication
print('\n')
print('-- Matrix sum and multiplication --')

A = numpy.ones((3,3))
B = numpy.ones((3,3))

print(A*B)

# Inverse
print('\n')
print('-- Matrix Inverse --')

# Splitting and replicating matrixes
print('\n')
print('-- Splitting and replicating Matrixes --')

## Display a error message with error code
print('\n')
print('-- Message with error code --')

errorCode = 72
print('Error code: ', errorCode)

## Generate random vectors with gaussian clouds
print('\n')
print('-- Generate random clusters --')

N = 2			# No of dimensions
K = 20			# No of classes

sigma = 0.25	# Gaussian variance, sigma²
L = 100			# Data size

dataSize = K*L 		# Dataset size

T = numpy.random.randn(N, K)
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

Y_unwrap = -1*numpy.ones(Y.shape)
Y_unwrap[:,targetIndex] = 1

print('Target classes shape: ', targetClasses.shape)
print('Target labels shape: ', targetLabels.shape)
print('Y_unwrap shape: ', Y_unwrap.shape)

print('\n')
print('-- Plot a  function --')

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
xIndex = numpy.random.permutation(numpy.arange(m))

x_train = X[:,xIndex[:dataSize*trainRatio]]
x_test = X[:,xIndex[dataSize*trainRatio:dataSize*(trainRatio+testRatio)]]
x_val = X[:,xIndex[dataSize*(trainRatio+testRatio):end]]

# Input and one hidden layer
model.add(Dense(units=10, input_dim=2))
model.add(Activation('sigmoid'))

# Output layer
model.add(Dense(units=2))
model.add(Activation('softmax'))

# Configure optimizer
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=learningRate, momentum=0.9, nesterov=True))

#model.fit(x_train, y_train, epochs=5, batch_size=10)

#metrics = model.evaluate(x_test, y_test, batch_size=20)