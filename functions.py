import numpy as np

class Classifier( object ):
	
	def __init__( self ):
		print "starting classifier"

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


# Testing classifier class
x = np.array([[1,2,3,4], [23, 13, 7, 5],[101, 201, 301, 401]])
y = np.array([[1,1,3,4], [11,22,33,44]])
w = np.array([[10],[20],[30]])


test = Classifier()

newX, newY =  test.shuffle( x, w )
test.SGD( x, w, y )

'''
print "reporting classification error"
err = test.classificationError( x, y )
print err
'''
