import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

class Model( object ):
	
	def __init__( self ):
		print "starting model"

	'''
	Function name: SGD()
	Function description: 
	'''
	def SGD( self, X, Y, W, learningRate = 0.00005 ):
		print "starting sgd"
		newX, newY = self.shuffle( X, Y )
	

	'''
	Function name: shuffle()
	Function description: shuffles the input data into random ordering
	Parameters:
		X -- feature vectors (assumes to be row vectors)
		Y -- label vectors (assumes to be column vector)
	Return values:
		shuffledX -- feature vectors into randomized order
		shuffledY -- label vectors ordered corresponding to feature vectors
	'''
	def shuffle( self, X, Y ):
		M = X

		M = np.append( M, Y, axis = 1 )
		np.random.shuffle( M )

		Xshape = X.shape
		newY = M[ :, -1 ]
		newX = M[ :, 0:Xshape[1] ]

		return newX, newY


	'''
	Function name: classificationError()
	Function description: this method provides the classification error for the
		classifier
	Parameters:
		trueLabel -- vector of true class labels
		classificationLabels -- classs labels as determined by the classifier
	Return value:
		err -- classification error for the classifier
	'''	
	def classificationError( self, trueLabel, classificationLabels ):
		trueLabelShape = trueLabel.shape
		classificationLabelShape = classificationLabels.shape
	
		if trueLabelShape != classificationLabelShape: 
			print "Incorrect input dimension, cannot generate classification error"
			print "Exiting..."
	
			return -1
	
		err = trueLabel == classificationLabels
		err = np.sum( err ) / float( trueLabelShape[0] )

		return ( 1 - err )	

class Classifier( object ):

	def __init__( self ):
		print "starting classifier"

			
	'''
	Xtrain -- row vector containing feature data
	'''
	def KMeans( self, Xtrain, K ):
		print "Starting K-Means"

		Xshape = Xtrain.shape
	
		# initialize cluster centers by randomly picking points from the data
		randIndex = np.random.permutation( Xshape[0] )
		randIndex = np.reshape( randIndex, ( Xshape[0], 1) )
		means = Xtrain[ randIndex[0:K], :]

		oldMeans = means	
		maxIters = 30000

		for i in range( maxIters ):
			distMatrix = self.squareDistanceMetric( Xtrain, means )
			rank = self.determineRank( distMatrix )
			
			oldMeans = means
			means = self.recalcMeans( Xtrain, rank )

			if np.linalg.norm( (oldMeans - means) ) < 0.0000001:
				print "i = ", i
				break	

	def plotCurrent( self, Data, Ranks, Means ):
		dataShape = Data.shape
		N = dataShape[0]
		D = dataShape[1]

		meanShape = Means.shape
		K = meanShape[0]

		# clear old plot
		plt.clf()
		figure( 1 )
		
	
	def recalcMeans( self, Data, Ranks ):
		dataShape = Data.shape
		rankShape = Ranks.shape

		N = dataShape[0]
		D = dataShape[1]
		K = rankShape[1]

		Ksum = np.sum( Ranks, axis = 0 )
		means = np.zeros( ( K, D ) )

		for i in range( N ):
			for j in range( K ):
				if Ranks[ i, j ] == 1:
					means[ j, : ] = means[ j, : ] + Data[ i, : ]

		for i in range( K ):
			if Ksum[ i ] != 0:
				means[ i, : ] =np.divide( means[ i, : ], Ksum[ i ] )

		return means

	def determineRank( self, distMatrix  ):
		distShape = distMatrix.shape
		N = distShape[0]
		K = distShape[1]

		index = np.argmin( distMatrix, axis = 1)
		index = np.reshape( index, ( N, 1 ) )
		rank = np.zeros( ( N, K ) )
		
		for i in range( N ):
			rank[ i, index[ i, 0 ] ] = 1

		return rank
	
	'''
	Assign each data vector to closest mu vector
	- do this by first calculating a squared distance matrix where the n,k entry
		contains the squared distance from the nth data vector to the kth mu vector
	'''
	def squareDistanceMetric( self, Data, Centers ):
		dataShape = Data.shape
		N = dataShape[0]

		K = Centers.shape
		K = K[0]

		sqDistance = np.zeros( ( N, K ) )
		for i in range( K ):
			for j in range( N ):
				# square distance in the jth cluster
				diff = Data[ j, : ] - Centers[ i, : ]
				diff = np.multiply( diff, diff )
				sqDistance[ j, i ] = np.sum( diff )
		
		return sqDistance
	
# Testing classifier class
x = np.array([[1,2,3,4], [23, 13, 7, 5],[101, 201, 301, 401]])
y = np.array([[1,1,3,4], [11,22,33,44]])
w = np.array([[10],[20],[30]])

data = np.loadtxt( 'Q4_data.txt',
									converters = { -1 : lambda s: { b'Iris-virginica': 0,
																									b'Iris-versicolor': 1}[s]})

x = data[ :, 0: 4 ]
y = data[ :, 4 ]
Yshape = y.shape
y = np.reshape(y , ( Yshape[0], 1 ))


Xtest = x[ 0:15, : ]
Ytest = y[ : 15 ] 
Ytest = np.append( Ytest, y[50:65] )
Xtest = np.append( Xtest, x[50:65,:], axis = 0 )

Xtrain = x[15:50, :]
Xtrain = np.append( Xtrain, x[65:100, :], axis = 0 )
Ytrain = y[15:50]
Ytrain  = np.append( Ytrain, y[65:100] )



test = Model()
newX, newY =  test.shuffle( x, y )

K = 5
test2 = Classifier()
test2.KMeans( Xtrain, K )
'''
print "reporting classification error"
err = test.classificationError( x, y )
print err
'''
