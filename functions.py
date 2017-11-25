import numpy as np

class Model( object ):
	
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

class Classifier( object ):
	''' Hopfield Network '''
	def hopfieldNetwork():
		print "Starting Hopfield Network"
	
		V = np.array( [ [ -1, 1, 1, -1, 1 ], [ 1, -1, 1, -1, 1 ] ] )

		W = generateWeightMatrix( V.T )

		V = np.array( [ [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ 1 ] ] )
	
		order = np.array( [ 3, 1, 5, 2, 4 ] )
		print "visiting pattern = ", order
		print "started with = "
		print V
		U, updateHistory  = hopfieldNetworkAlgorithm( W, V, order )
		print "updateHistory = "
		print updateHistory
		print "ended with = "
		print U
	
		print "\n"
		V = np.array( [ [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ 1 ] ] )
		order = np.array( [ 2, 4, 3, 5, 1 ] )
		print "visiting pattern = ", order
		print "started with = "
		print V
		U, updateHistory  = hopfieldNetworkAlgorithm( W, V, order )
		print "updateHistory = "
		print updateHistory
		print "ended with = "
		print U

	'''
	Function Name: hopfieldNetworkAlgorithm()
	Function description: this function perform the algorithm for hopfield
		network algorithm
	Parameters:
		W -- weight matrix of the network
		V -- initial state of the network
		order -- vector specifying the visiting order of each nodes
							by default, its sequential visiting order
	Return values:
		U -- end state of the hopfield network
		updateHistory -- matrix whose columns represent the state of hopfield
			network at each iteration
	'''	
	def  hopfieldNetworkAlgorithm( W, V, order = None ):
		U = V
		Vshape = V.shape
		updateHistory = U

		if order is None:
			order = range( Vshape[ 0 ] )
	
		error = True
		count = 0

		while error:
			error = False
			for i in order:
				temp = evolve( W, U, i )
				count += 1
				if U[ i - 1 ] != temp:
					count = 0
					error = True
				U[ i - 1 ] = temp
				updateHistory = np.append( updateHistory, U, axis = 1 )
				if count == Vshape[0]:
					break

		return U, updateHistory

	'''
	Function name: evolve()
	Function description: this function determines the next state of a
		given neuron for the hopfield network algorithm
	Parameters:
		W -- weight matrix of the hopfield network
		X -- current state of the hopfield network
		position -- node whose state needs to be updated
	Return values:
		-1 -- if the next state of the neuron is OFF
		+1 -- if the next state of the neuron is ON
	'''	
	def evolve( W, X, position ):	
		newX = np.dot( W[ position - 1, : ], X )
		if newX < 0:
			return -1
		else:
			return 1

	'''
	Function name: generateWeightMatrix()
	Functiion description: this function generates the weight matrix for
		the hopfield network
	Parameters:
		X -- matrix whose columns represents the patterns that needs to be
					memorized
	Return value:
		W -- weight matrix of the hopfield network
	'''
	def generateWeightMatrix( X ):
		XShape = X.shape
		W = np.matmul( X, X.T )/ float( XShape[0] )
		W = W - ( ( float( XShape[1] ) / float( XShape[0] ) ) * np.identity( XShape[0] ) )

		return W


# Testing classifier class
x = np.array([[1,2,3,4], [23, 13, 7, 5],[101, 201, 301, 401]])
y = np.array([[1,1,3,4], [11,22,33,44]])
w = np.array([[10],[20],[30]])

data = np.loadtxt( 'Q4_data.txt',
									converters = { -1 : lambda s: { b'Iris-virginica': 0,
																									b'Iris-versicolor': 1}[s]})

print data

test = Model()

newX, newY =  test.shuffle( x, w )
test.SGD( x, w, y )

'''
print "reporting classification error"
err = test.classificationError( x, y )
print err
'''
