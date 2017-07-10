import numpy as numpy
import numpy.matlib as matlib
import matplotlib.pyplot as pyplot

class Dataset:
	"""docstring for Dataset"""
	def __init__(self, dim, clusterSize, numClasses, variance):
		self.dim = dim
		self.clusterSize = clusterSize
		self.numClasses = numClasses
		self.variance = variance
		self.dataSize = clusterSize*numClasses
		
	def generate(self):
		# Generate random data
		self.T = numpy.random.randn(self.dim, self.numClasses) # Cluster centers
		self.X = matlib.repmat(self.T, 1, self.clusterSize) + self.variance*numpy.random.randn(self.dim, self.dataSize)
		self.Y = matlib.repmat(range(self.numClasses),1,self.clusterSize)
		return self.X, self.Y, self.T
		
	def unwrap(self):
		# Unwrap labels
		self.targetClasses = numpy.random.permutation(numpy.arange(self.numClasses))[:numpy.floor(self.numClasses/2).astype(int)]	# K/2 array of target classes

		self.targetIndex = numpy.in1d(self.Y,self.targetClasses) # Indexes of I/O pairs belonging to target classes
		self.targetLabels = self.X[:,self.targetIndex]			  # Input elements belonging to target classes

		Y_unwrap = numpy.zeros(self.Y.shape)
		Y_unwrap[:,self.targetIndex] = 1

		return Y_unwrap

	def plotData(self):
		# Plot the data
		X = self.X
		Y = self.X
		T = self.T
		targetClasses = self.targetClasses 
		targetLabels =  self.targetLabels

		pyplot.figure(figsize=(10,10))
		pyplot.plot(X[0,:], X[1,:], 'b+')
		pyplot.plot(T[0,:], T[1,:], 'ko', markersize=8, markerfacecolor='w', markeredgewidth=2)
		pyplot.title('Random Clusters')

		pyplot.plot(targetLabels[0,:], targetLabels[1,:], 'r+')
		pyplot.plot(T[0,targetClasses], T[1,targetClasses], 'kx', markersize=13, markeredgewidth=2)
		return