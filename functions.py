import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from random import *

class Model( object ):
	
	def __init__( self ):
		print "starting model"

	def kFoldCrossValidation( self, X, Y, num_folds = 5, model = None ):
		print "starting K-Fold cross validation: K = ", num_folds

		# Randomizing data
		X, Y = self.shuffle( X, Y )

		Xtrain_shape = X.shape

		for i in range( num_folds ):
			# getting the lower and upper bound for the new dataset
			lower = i * Xtrain_shape[ 0 ] / num_folds
			upper = lower + ( Xtrain_shape[ 0 ] / num_folds )

			# new testing set
			newX_test = X[ lower : upper ]
			newY_test = Y[ lower : upper ]

			# new training set
			newX_train = np.delete( X, range( lower, ( upper + i ) ), 0 )
			newY_train = np.delete( Y, range( lower, ( upper + i ) ), 0 )

	

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
		Y = np.reshape( Y, ( Y.shape[ 0 ], 1 ) )
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

	'''
	Function name: meanSquareError()
	Function description: this method provides the mean square error for the
		classifier
	Parameters:
		trueLabel -- vector of true class labels
		classificationLabels -- classs labels as determined by the classifier
	Return value:
		err -- mean square error for the classifier
	'''	
	def meanSquareError( self, trueLabel, classificationLabels ):
		trueLabelShape = trueLabel.shape
		classificationLabelShape = classificationLabels.shape

		if trueLabelShape != classificationLabelShape:
			print "Incorrect input dimension, cannot generate MSError"
			print "Exiting..."

			return -1
		err = self.classificationError( trueLabel, classificationLabels )
		return np.sqrt( err )

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
class Classifier( object ):

	def __init__( self ):
		print "starting classifier"

	def relu( self, x ):
		relu = x * ( x > 0 ).astype( float )
		print relu
		return relu		
	

	def K_Fold_Cross_Validation_KMeans( self, XTrain, Ytrain, num_folds = 5 ):
		print "starting K-Fold cross validation: num_folds = ", num_folds

		model = Model()
		# Randomizing data
		X, Y = model.shuffle( Xtrain, Ytrain )

		Xtrain_shape = X.shape

		K = range( 2, 20 )
		print "K = ", K

		Y = self.one_hot_encoder( Y, num_classes = 2 )
		
		for j in K:
			for i in range( num_folds ):
				# getting the lower and upper bound for the new dataset
				lower = i * Xtrain_shape[ 0 ] / num_folds
				upper = lower + ( Xtrain_shape[ 0 ] / num_folds )

				# new testing set
				newX_test = X[ lower : upper ]
				newY_test = Y[ lower : upper ]

				# new training set
				newX_train = np.delete( X, range( lower, ( upper + i ) ), 0 )
				newY_train = np.delete( Y, range( lower, ( upper + i ) ), 0 )

				# running K-Means on training dataset
				ranks, means = self.KMeans( newX_train, K = j )
				classLabels = labelCluster( ranks, newX_train, K = j )

				# Classification of testing dataset
				dist = self.squareDistanceMetric( newX_test, means )
				testRank = self.determineRank( dist )
				clusterLabel = self.labelCluster( testRank, newY_test, K = j )
			
				break

			break


	def labelCluster( self, Ranks, labels, K ):
		clusterLabel = np.random.rand( 1, K )
		
		rankShape = Ranks.shape
	
		temp = np.zeros( ( 1, labels.shape[ 1 ] ) )

		for i in range( K ):
			for j in range( rankShape[ 0 ] ):
				if Ranks[ j, i ] == 1:
					temp += labels[ j, : ]

			clusterLabel[ 0, i ] = np.argmax( temp )

		print "clusterLabel = ", clusterLabel
		
		return clusterLabel

	'''
	Function name: one_hot_encoder()
	Function description: this function encodes the data vector via one-hot encoding
	Parameters:
		y -- data vector to encode ( shape = ( N, 1 ) )
		num_classes -- number of classes (optional)
	Return value:
		one_hot_encoder -- matrix composed to one-hot vector of y
	'''
	def one_hot_encoder( self, y, num_classes = None ):
		y = np.array( y, dtype = 'int' )
		input_shape = y.shape

		if input_shape and input_shape[ -1 ] == 1 and len( input_shape ) > 1:
			input_shape = tuple( input_shape[ : -1 ] )

		y = y.ravel()

		if not num_classes:
			num_classes = np.max( y ) + 1

		n = y.shape[ 0 ]
		
		categorical = np.zeros( ( n, num_classes ) )
		categorical[ np.arange( n ), y ] = 1
		output_shape = input_shape + ( num_classes, )
		categorical = np.reshape( categorical, output_shape )
		return categorical
	
	'''
	Function name: KMeans()
	Function description: thie method performs the K-means clustering algorithm
		on the input dataset
	Parameters:
		X -- feature vecctor, where each data point is a row vector
		K -- number of clusters, (by default, K = 2 )
	Return value:
		ranks -- matrix containing which cluster each datapoint belongs to,
							uses one-hot encoding for the cluster
	'''
	def KMeans( self, X, K = None ):
		if K == None:
			K = 2

		Xshape = X.shape
	
		# initialize cluster centers by randomly picking points from the data
		randIndex = np.random.permutation( Xshape[0] )
		randIndex = np.reshape( randIndex, ( Xshape[0], 1) )
		means = X[ randIndex[0:K], :]

		oldMeans = means	
		maxIters = 30000

		for i in range( maxIters ):
			distMatrix = self.squareDistanceMetric( X, means )
			rank = self.determineRank( distMatrix )
			
			oldMeans = means
			means = self.recalcMeans( X, rank )

			if np.linalg.norm( (oldMeans - means) ) < 0.0000001:
				#print "i = ", i
				break	
		
		return rank, means

	'''
	Function name: recalcMeans()
	Function description: this method computes the cluster means
	Parameters:
		Data -- datapoints, as row vectors
		Ranks -- matrix depicting which cluster each datapoints belong to
	Return value:
		means -- means for each cluster
	'''	
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

	'''
	Function name: determinRank()
	Function description: this method classifies each datapoint into one of
		the clusters
	Parameters:
		distMatrix -- matrix containing the distance of each datapoint from each
			of the cluster
	Return value:
		rank -- matrix containing which cluster each datapoint belongs to,
							uses one-hot encoding for the cluster
	'''
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
	Function name: sqaureDistanceMetric()
	Function description: this method calculates the square of the distance of
		each datapoint from each cluster
	Parameters:
		Data -- datapoints, each datapoint is a row vector
		Centers -- means for each cluster
	Return value:
		sqDist -- matrix containing the square distance of each datapoint from
							each cluster
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

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################	
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


'''
test = Model()
newX, newY =  test.shuffle( x, y )

x = np.array([[1, 2, 3, 4, -1, -2]])
K = 2
test2 = Classifier()
rank = test2.KMeans( Xtrain, K )
print "rank = ", rank.shape
test2.relu( x )
'''
testKFold = Classifier()
print "Xshape = ", Xtrain.shape
K = 5
testKFold.K_Fold_Cross_Validation_KMeans( Xtrain, Ytrain, K )
