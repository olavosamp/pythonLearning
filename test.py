import dataset

N = 2			# No of dimensions
L = 100			# Data size
K = 20			# No of classes
sigma = 0.25

dataSet = dataset.Dataset(N, L, K, sigma)

dataSet.generate()

print(dataSet.X.shape)

dataSet.unwrap()
dataSet.plotData()