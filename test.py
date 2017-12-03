from sklearn.datasets import fetch_mldata
import sklearn.utils as rnd
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

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

'''
Feedforward Neural Network algorithm with gradient descent
'''
def feed_forward_neural_network():
	print "Starting Feed-Forward Neural Network"
	
	''' loading data'''
	# Iris-versicolor = 1
	# Iris-virginica = -1
	data = np.loadtxt( 'Q4_data_modified.txt', dtype = 'float' )
	x = data[ :, 0 : 4 ]
	y = data[ :, 4 ]

	Xshape = x.shape	
	x = np.append( x, np.tile( 1, ( Xshape[0], 1 ) ), axis = 1 )

	Xtrain, Ytrain, Xtest, Ytest = splitData( x, y )
	Xtrain = np.transpose( Xtrain )
	Xtest = np.transpose( Xtest )

	''' iniatilizing weights '''
	W1 = np.ones( ( 5, 2 ) )
	W2 = np.ones( ( 3, 1 ) )
	stepSize = 0.0001

	newW1 = 0 * W1
	newW2 = 0 * W2

	lossErr = np.array( [ 1 ] )	

	
	for i in range( 10000 ):
		dW1 = gradientW1( Xtrain, Ytrain, W1, W2 )
		dW2 = gradientW2( Xtrain, Ytrain, W1, W2 )
		
		#print dW1
		W1 = W1 - stepSize * dW1
		W2 = W2 - stepSize * dW2
		
		# finding training error
		temp = classify( Xtrain, W1, W2 )
		lossErr = np.append( lossErr, loss( Ytrain, temp ) )
		
	print "Training error = "
	print "Testing error = "

	plt.plot( lossErr )
	plt.xlabel( "iteration" )
	plt.ylabel( 'loss' )
	plt.title( 'loss vs. iteration' )
	plt.show()

'''
Function name: gradientW1()
Function description: this function find the gradient of f(X) with respect
	to W1, for feedforward neural network
Parameters:
	x -- feature vectors used during training
	y -- label vector used during training
	W1 -- weight matrix 1
	W2 -- weight mtarix 2
Return value:
	gradW1 -- gradient of f(X) with respect to W1
'''

def gradientW1( x, y, W1, W2 ):
	L = gradientLoss( x, y, W1, W2 )
	L = L.T

	W1_1 = W1[ :, 0 ]
	W1_1 = np.vstack( W1_1 )
	W1_2 = W1[ :, 1 ]
	W1_2 = np.vstack( W1_2 )

	q1 = np.dot( np.transpose( W1_1 ), x )	# ( 1, 70 )
	q2 = np.dot( np.transpose( W1_2 ), x )	# ( 1, 70 )

	dq1_dW1 = sigmoid( q1 ) 	# ( 1, 70 )
	dq1_dW1 = dq1_dW1 * ( 1 - dq1_dW1 ) # ( 1, 70 )

	dq2_dW1 = sigmoid( q2 )	# ( 1, 70 )
	dq2_dW1 = dq2_dW1 * ( 1 - dq2_dW1 )

	df_dW1_1 = L * ( dq1_dW1 * x )
	df_dW1_1 = np.sum( df_dW1_1 , axis = 1 )	# ( 5, 1 )
	df_dW1_1 = np.vstack( df_dW1_1 )

	df_dW1_2 = L * ( dq2_dW1 * x )
	df_dW1_2 = np.sum( df_dW1_2, axis = 1 )	# ( 5, 1 )
	df_dW1_2 = np.vstack( df_dW1_2 )

	grad = np.append( df_dW1_1, df_dW1_2, axis = 1 ) 	# ( 5, 2 )
	return grad

'''
Function name: gradientW2()
Function description: this function find the gradient of f(X) with respect
	to W2, for feedforward neural network
Parameters:
	x -- feature vectors used during training
	y -- label vector used during training
	W1 -- weight matrix 1
	W2 -- weight mtarix 2
Return value:
	gradW2 -- gradient of f(X) with respect to W2
'''
def gradientW2( x, y, W1, W2 ):
	Xshape = x.shape
	Yshape = y.shape
	
	L = gradientLoss( x, y, W1, W2 )

	q = np.dot( np.transpose( W1 ), x )
	q = sigmoid( q )
	q = np.append( q, np.tile( 1, ( 1, Xshape[1] ) ), axis = 0 )
		
	p = np.dot( np.transpose( W2 ), q )
	p = sigmoid( p )
	p = np.multiply( p, ( 1 - p ) )

	grad = p * q
	return np.dot( grad, L )

'''
Function name: gradientLoss()
Function description: this function finds the gradient of the loss function
	with respect to f(X), for feedforward neural network
Parameters:
	x -- feature vector used during training
	y -- label vector for the training dataset
	W1 -- weight matrix 1
	W2 -- wwight matrix 2
Return value:
	ynew -- gradient of the loss function with respect to f(X)
'''
def gradientLoss( x, y, W1, W2 ):
	f = classify( x, W1, W2  )
	y = np.vstack( y )
	return ( f - y ) / ( f * ( 1 - f ) )

'''
Returns classification error
'''
def loss( trueLabels, classificationLabels ):
	classSize = trueLabels.shape
	
	trueLabels = np.vstack( trueLabels )
	temp = inRange( classificationLabels )
	
	err = trueLabels - temp	
	err = np.absolute( err )
	
	return np.sum( err ) / ( 2 * float(classSize[0]) )

'''
Function name: classif()
Function description: this method classifies input feature vectors into
	one of the two possible class for feedforward neural network
Parameters:
	x -- vector containing all the features/data vector
	W1 -- weight matrix 1
	W2 -- weight matrix 2
Return value:
	y -- class vector
'''
def classify( x, W1, W2 ):
	Xshape = x.shape
	
	q = np.dot( np.transpose( W1 ), x )
	q = sigmoid( q )
	temp = np.append( q, np.tile( 1, ( 1, Xshape[1] ) ), axis = 0 )

	p = np.dot( np.transpose( W2 ), temp )
	p = sigmoid( p )

	return np.transpose( p )

'''
Function name: inRange()
Function description: this function applies the linear decision boundary on
	the indput vector
Parameter:
	X -- input vector to transform
Return value:
	newX -- new vector with the applied decision boundary on each entry
'''	
def inRange( x ):
	Xshape = x.shape
	newX = np.array( x )
	
	for i in range( Xshape[0] ):
		if x[i] < 0.5:
			newX[i] = -1
		else:
			newX[i] = 1

	return newX

'''
Function name: sigmoid()
function description: this function applies the sigmoid transformation on each
	dimension of input vector
Parameter:
	Z -- the vector that needs to be transform
Return value: sigmoid transformation of the input vector
'''
def sigmoid( Z ):
	return 1 / ( 1 + np.exp( -1 * Z ) )	

'''
Function name: splitData()
Function Description: this method splits the input data into training set and
		testing set for feedforward neural network algorithm
Parameters:
	x -- input feature vectors
	y -- input label vectors
Return value:
	Xtrain -- training set of feature vectors
	Ytrain -- training set of label vectors
	Xtest -- testing set of feature vectors
	Ytest -- testing set of label vectors
'''
def splitData( x, y ):
	Xtest = x[ 0:15, : ]	
	Ytest = y[ 0:15 ]
	Ytest = np.append( Ytest, y[50:65] )
	Xtest = np.append( Xtest, x[50:65, :], axis = 0)

	Xtrain = x[ 15:50, : ]
	Xtrain = np.append( Xtrain, x[65:100, :], axis = 0 )
	Ytrain = y[ 15:50 ]
	Ytrain = np.append( Ytrain, y[65:100] )

	return Xtrain, Ytrain, Xtest, Ytest

'''
This function perform the multilayer perceptron algorithm 
on MNIST datatset
'''
def multilayerPerceptron():
	print "Starting Multilayer Perceptron Algorithm"

	''' Fetching Data '''
	mnist = fetch_mldata( "MNIST original" )
	data = mnist.data
	target = mnist.target

	''' Shuffling data '''
	data = rnd.shuffle( data, random_state = 1 )
	target = rnd.shuffle( target, random_state = 1 )

	''' splitting data into training and testing set '''
	trainX = data[ 0:60000,:]
	trainY = target[ 0:60000,]

	testX = data[  60000:70000,:]
	testY = target[ 60000:70000,]

	''' testcase 1.1 '''
	testMLPClassifier(  trainX, trainY, testX, testY, 
		title = 'MLPClassifier, 60 hidden units, Adam solver', 
		filePath = './Cogs181_Q4_1_1.png', 
		layerSize = 60, activationFunction = 'relu', solverFun = 'adam', )

	''' testcase 1.2 '''
	testMLPClassifier(  trainX, trainY, testX, testY, 
		title = 'MLPClassifier, 60 hidden units, SGD solver', 
		filePath = './Cogs181_Q4_1_2.png', 
		layerSize = 60, activationFunction = 'relu', solverFun = 'sgd', )

	''' testcase 2.1 '''
	testMLPClassifier(  trainX, trainY, testX, testY, 
		title = 'MLPClassifier, 20 hidden units, Adam solver', 
		filePath = './Cogs181_Q4_2_1.png', 
		layerSize = 20, activationFunction = 'relu', solverFun = 'adam', )

	''' testcase 2.2 '''
	testMLPClassifier(  trainX, trainY, testX, testY, 
		title = 'MLPClassifier, 50 hidden units, Adam solver', 
		filePath = './Cogs181_Q4_2_2.png', 
		layerSize = 50, activationFunction = 'relu', solverFun = 'adam', )

	''' testcase 2.3 '''
	testMLPClassifier(  trainX, trainY, testX, testY, 
		title = 'MLPClassifier, 100 hidden units, Adam solver', 
		filePath = './Cogs181_Q4_2_3.png', 
		layerSize = 100, activationFunction = 'relu', solverFun = 'adam', )
	
	''' testcase 3.1 '''
	testMLPClassifier(  trainX, trainY, testX, testY, 
		title = 'MLPClassifier, 60 hidden units, Adam solver', 
		filePath = './Cogs181_Q4_3_1.png', 
		layerSize = 60, activationFunction = 'relu', solverFun = 'adam', )

	''' testcase 3.2 '''
	testMLPClassifier(  trainX, trainY, testX, testY, 
		title = 'MLPClassifier, 2 hidden layers, 30 hidden units each, Adam solver', 
		filePath = './Cogs181_Q4_3_2.png', 
		layerSize = (30,30), activationFunction = 'relu', solverFun = 'adam', )

	'''testcase 3.3 '''
	testMLPClassifier(  trainX, trainY, testX, testY, 
		title = 'MLPClassifier, 3 hidden layers, 20 hidden units each, Adam solver', 
		filePath = './Cogs181_Q4_3_3.png', 
		layerSize = (20,20,20), activationFunction = 'relu', solverFun = 'adam', )


'''
Function name: plotLossCurve()
Function description: this function plots the loss curve for MLPClassifier
Parameters:
	classifier -- MLPClassifier that has already been trained on data
	title -- title of the generated plot
	filePath -- complete file path to the location as to where to save the generated plot
Return value: none
'''
def plotLossCurve( classifier, title, filePath ):
	plt.clf() # clear old plot
	plt.plot( classifier.loss_curve_ ) # plot new plot
	plt.ylabel( 'loss' )
	plt.xlabel( 'iteration' )
	plt.title( title )
	#plt.savefig( filePath )
	#plt.show()

'''
Function name: testMLPClassifier()
Function description: this method performs the MLP algorithm on training
	data and perform classification on test data to test for accuracy
Parameters:
	Xtrain -- training feature vectors
	Ytrain -- training label vector
	Xtest --	testing feature vectors
	Ytrest -- testing label vector
	title -- description of classifier specification
	filePath -- complete file path to location as to where to save the generated plot
	layerSize -- size of hidder layers (default = 60), int or tuple
	activationFunction -- activation function of MLPClassifier (default = 'relu')
	solverFun -- slover function for MLPClassifier (default = 'adam')
Return value: None
'''
def testMLPClassifier( Xtrain, Ytrain, Xtest, Ytest, title, filePath,
	 layerSize = 60, activationFunction = 'relu', solverFun = 'adam' ):

	''' Running classifier '''
	mlp = MLPClassifier( hidden_layer_sizes = layerSize, activation = activationFunction, solver = solverFun )
	mlp.fit( Xtrain, Ytrain )
	print title
	score = mlp.score( Xtest, Ytest )
	print "score = ", score

	''' Plotting loss curve '''
	plotLossCurve( mlp, title, filePath )
	
#hopfieldNetwork()
#feed_forward_neural_network()
#multilayerPerceptron()
