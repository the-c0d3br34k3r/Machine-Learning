import numpy as np
import math
from scipy.io import loadmat
from scipy.optimize import minimize
import sklearn.svm as SVM
import pickle

def preprocess():
	"""
	 Input:
	 Although this function doesn't have any input, you are required to load
	 the MNIST data set from file 'mnist_all.mat'.

	 Output:
	 train_data: matrix of training set. Each row of train_data contains
	   feature vector of a image
	 train_label: vector of label corresponding to each image in the training
	   set
	 validation_data: matrix of training set. Each row of validation_data
	   contains feature vector of a image
	 validation_label: vector of label corresponding to each image in the
	   training set
	 test_data: matrix of training set. Each row of test_data contains
	   feature vector of a image
	 test_label: vector of label corresponding to each image in the testing
	   set
	"""

	mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

	n_feature = mat.get("train1").shape[1]
	n_sample = 0
	for i in range(10):
		n_sample = n_sample + mat.get("train" + str(i)).shape[0]
	n_validation = 1000
	n_train = n_sample - 10 * n_validation

	# Construct validation data
	validation_data = np.zeros((10 * n_validation, n_feature))
	for i in range(10):
		validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

	# Construct validation label
	validation_label = np.ones((10 * n_validation, 1))
	for i in range(10):
		validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

	# Construct training data and label
	train_data = np.zeros((n_train, n_feature))
	train_label = np.zeros((n_train, 1))
	temp = 0
	for i in range(10):
		size_i = mat.get("train" + str(i)).shape[0]
		train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
		train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
		temp = temp + size_i - n_validation

	# Construct test data and label
	n_test = 0
	for i in range(10):
		n_test = n_test + mat.get("test" + str(i)).shape[0]
	test_data = np.zeros((n_test, n_feature))
	test_label = np.zeros((n_test, 1))
	temp = 0
	for i in range(10):
		size_i = mat.get("test" + str(i)).shape[0]
		test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
		test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
		temp = temp + size_i

	# Delete features which don't provide any useful information for classifiers
	sigma = np.std(train_data, axis=0)
	index = np.array([])
	for i in range(n_feature):
		if (sigma[i] > 0.001):
			index = np.append(index, [i])
	train_data = train_data[:, index.astype(int)]
	validation_data = validation_data[:, index.astype(int)]
	test_data = test_data[:, index.astype(int)]

	# Scale data to 0 and 1
	train_data /= 255.0
	validation_data /= 255.0
	test_data /= 255.0
	return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
	"""
	blrObjFunction computes 2-class Logistic Regression error function and
	its gradient.

	Input:
		initialWeights: the weight vector (w_k) of size (D + 1) x 1
		train_data: the data matrix of size N x D
		labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

	Output:
		error: the scalar value of error function of 2-class logistic regression
		error_grad: the vector of size (D+1) x 1 representing the gradient of
					error function
	"""
	train_data, labeli = args

	n_data = train_data.shape[0]
	n_features = train_data.shape[1]
	error = 0
	error_grad = np.zeros(n_features + 1)

	##################
	# YOUR CODE HERE #
	##################
	# HINT: Do not forget to add the bias term to your input data
	# bias =  np.ones((n_data,), dtype=np.int)
	labeli = labeli.reshape((labeli.shape[0],))
	train_data_with_bias = np.empty([n_data, n_features + 1])
	for i in range(len(train_data)):
		train_data_with_bias[i] = np.append(train_data[i], 1)
	# train_data_with_bias = [list(i) for i in zip(train_data,bias)]
	# train_data_with_bias = np.array(zip(train_data,bias))
	# print(train_data_with_bias)
	# print('Train data size - '+str(train_data_with_bias.shape)+', ')
	initialWeights = initialWeights.reshape((initialWeights.shape[0], 1))
	dotProduct = np.dot(train_data_with_bias, initialWeights)
	theta_n = sigmoid(dotProduct)
	# print('Theta_n - ' + str(theta_n.shape))
	# print('Calculate error start')
	for i in range(n_data):
		error += labeli[i] * math.log(theta_n[i]) + (1 - labeli[i]) * math.log(1 - theta_n[i])
	for i in range(n_data):
		error_grad = np.sum([error_grad, np.multiply(train_data_with_bias[i], (theta_n[i] - labeli[i]))], axis=0)
	# print('Calculate error end')
	error = -error / n_data
	error_grad = error_grad / float(n_data)
	print('Hello')
	return error, error_grad


def blrPredict(W, data):
	"""
	blrObjFunction predicts the label of data given the data and parameter W
	of Logistic Regression

	Input:
		W: the matrix of weight of size (D + 1) x 10. Each column is the weight
		vector of a Logistic Regression classifier.
		X: the data matrix of size N x D

	Output:
		label: vector of size N x 1 representing the predicted label of
		corresponding feature vector given in data matrix
	"""
	lbl = np.zeros((data.shape[0], 1))
	n_data = data.shape[0]
	n_features = data.shape[1]
	##################
	# YOUR CODE HERE #
	##################
	# HINT: Do not forget to add the bias term to your input data
	# bias =  np.ones((n_data,),dtype=np.int)
	# data_with_bias = zip(data,bias)

	data_with_bias = np.empty([n_data, n_features + 1])
	print('Length of training data - ' + str(len(train_data)))
	for i in range(len(data)):
		data_with_bias[i] = np.append(data[i], 1)

	dot_product = np.dot(data_with_bias, W)
	lbl = np.argmax(dot_product, axis=1)
	lbl = lbl.reshape((lbl.shape[0], 1))
	return lbl


def mlrObjFunction(params, *args):
	"""
	mlrObjFunction computes multi-class Logistic Regression error function and
	its gradient.

	Input:
		initialWeights: the weight vector of size (D + 1) x 1
		train_data: the data matrix of size N x D
		labeli: the label vector of size N x 1 where each entry can be either 0 or 1
				representing the label of corresponding feature vector

	Output:
		error: the scalar value of error function of multi-class logistic regression
		error_grad: the vector of size (D+1) x 10 representing the gradient of
					error function
	"""
	iW = params.reshape((716,10));
	print('Params - '+str(iW.shape))
	train_data, labeli = args
	n_data = train_data.shape[0]
	n_feature = train_data.shape[1]
	error = 0.0
	error_grad = np.zeros((n_feature + 1, n_class))
#	labeli = labeli.reshape((labeli.shape[0],))
	train_data_with_bias = np.empty([n_data, n_feature + 1])
	for i in range(len(train_data)):
		train_data_with_bias[i] = np.append(train_data[i], 1)
	#initialWeights = initialWeights.reshape((initialWeights.shape[0], 1))
	dotProduct = np.dot(train_data_with_bias, iW)
	print(' - '+str(dotProduct.shape))
	print('theta calculated..')
	labeli = labeli.ravel();
	dotProduct = np.exp(dotProduct)
	for i in range(len(train_data)):
		dotProduct[i] = dotProduct[i]/np.sum(dotProduct[i])
#		print('theta - '+str(dotProduct.shape))
		error += math.log(dotProduct[i][int(labeli[i])])
		#interm = dotProduct[i]
		#interm[labeli[i]] = interm[labeli[i]] - 1
		error_grad = np.sum([error_grad, np.dot(np.transpose(train_data_with_bias[i].reshape((1,train_data_with_bias[i].shape[0]))),(dotProduct[int(labeli[i])] - 1).reshape((1,10)))], axis=0)
	#theta_n = sigmoid(dotProduct)
	#error_grad = error_grad.reshape((error_grad.shape[0],))
	##################
	# YOUR CODE HERE #
	##################
	# HINT: Do not forget to add the bias term to your input data
	return error, error_grad.flatten()


def mlrPredict(W, data):
	"""
	 mlrObjFunction predicts the label of data given the data and parameter W
	 of Logistic Regression

	 Input:
		 W: the matrix of weight of size (D + 1) x 10. Each column is the weight
		 vector of a Logistic Regression classifier.
		 X: the data matrix of size N x D

	 Output:
		 label: vector of size N x 1 representing the predicted label of
		 corresponding feature vector given in data matrix

	"""
	label = np.zeros((data.shape[0], 1))

	##################
	# YOUR CODE HERE #
	##################
	# HINT: Do not forget to add the bias term to your input data

	return label


"""
Script for Logistic Regression
"""
print('start')

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10
print('classes')

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))

print('Initializing Y')
for i in range(n_class):
	print(i)
	Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
'''
print('Train Label - ' + str(train_label.shape))
for i in range(n_class):
	labeli = Y[:, i].reshape(n_train, 1)
	args = (train_data, labeli)
	nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
	W[:, i] = nn_params.x.reshape((n_feature + 1,))
## Find the accuracy on Training Dataset
pickle.dump(W,open('params.pickle','wb'))
predicted_label = blrPredict(W, train_data)
print('Test')
print('Predicted Label - '+str(predicted_label.shape))
print('Train Label - '+str(train_label.shape))
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
#
## Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
'''
"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

linearSvm = SVM.SVC(cache_size=10000)
print("linearSvm fit start")
#labeli = labeli[0]
print('train_data - ' + str(train_data.shape));
print('train_label ravel - ' + str(train_label.shape));
print('validation_label - ' + str(validation_label.shape));
print('test_label - ' + str(test_label.shape));
print('train_label - ' + str(train_label.shape));
linearSvm.fit(train_data, train_label.ravel())
print("linearSvm fit done")
SVM.SVC(cache_size=10000, kernel='linear', verbose=True)
print("linearSvm predict start")
# Find the accuracy on Training Dataset
linearSvm_predicted_label = linearSvm.predict(train_data)
linearSvm_predicted_label = linearSvm_predicted_label.reshape((linearSvm_predicted_label.shape[0],1))

print('linearSvm_predicted_label - '+str(linearSvm_predicted_label.shape));
print('\n Training set Accuracy for linear SVM:' + str(100 * np.mean((linearSvm_predicted_label == train_label).astype(float))) + '%')

#Find the accuracy on Validation Dataset
linearSvm_predicted_label = linearSvm.predict(validation_data)
linearSvm_predicted_label = linearSvm_predicted_label.reshape((linearSvm_predicted_label.shape[0],1))
print('\n Validation set Accuracy for linear SVM:' + str(100 * np.mean((linearSvm_predicted_label == validation_label).astype(float))) + '%')

#Find the accuracy on Testing Dataset
linearSvm_predicted_label = linearSvm.predict(test_data)
linearSvm_predicted_label = linearSvm_predicted_label.reshape((linearSvm_predicted_label.shape[0],1))
print('\n Testing set Accuracy for linear SVM:' + str(100 * np.mean((linearSvm_predicted_label == test_label).astype(float))) + '%')

radialSvm = SVM.SVC()
radialSvm.fit(train_data, train_label.ravel())
SVM.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		decision_function_shape=None, degree=3, gamma=1.0, kernel='radial',
		max_iter=-1, probability=False, random_state=None, shrinking=True,
		tol=0.001, verbose=False)

# Find the accuracy on Training Dataset
radialSvm_predicted_label = radialSvm.predict(train_data)
radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
print('\n Training set Accuracy for radial 1.0 :' + str(
	100 * np.mean((radialSvm_predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
radialSvm_predicted_label = radialSvm.predict(validation_data)
radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
print('\n Validation set Accuracy for radial 1.0 :' + str(
	100 * np.mean((radialSvm_predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
radialSvm_predicted_label = radialSvm.predict(test_data)
radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
print('\n Testing set Accuracy for radial 1.0 :' + str(
	100 * np.mean((radialSvm_predicted_label == test_label).astype(float))) + '%')

radialSvm = SVM.SVC()
radialSvm.fit(train_data, train_label.ravel())
SVM.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		decision_function_shape=None, degree=3, gamma='auto', kernel='radial',
		max_iter=-1, probability=False, random_state=None, shrinking=True,
		tol=0.001, verbose=False)

# Find the accuracy on Training Dataset
radialSvm_predicted_label = radialSvm.predict(train_data)
radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
print('\n Training set Accuracy for radial auto : ' + str(
	100 * np.mean((radialSvm_predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
radialSvm_predicted_label = radialSvm.predict(validation_data)
radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
print('\n Validation set Accuracy for radial auto : ' + str(
	100 * np.mean((radialSvm_predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
radialSvm_predicted_label = radialSvm.predict(test_data)
radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
print('\n Testing set Accuracy for radial auto : ' + str(
	100 * np.mean((radialSvm_predicted_label == test_label).astype(float))) + '%')

for i in range(10, 100, 10):
	radialSvm = SVM.SVC()
	radialSvm.fit(train_data, train_label.ravel())
	SVM.SVC(C=i, cache_size=200, class_weight=None, coef0=0.0,
			decision_function_shape=None, degree=3, gamma='auto', kernel='radial',
			max_iter=-1, probability=False, random_state=None, shrinking=True,
			tol=0.001, verbose=False)

	# Find the accuracy on Training Dataset
	radialSvm_predicted_label = radialSvm.predict(train_data)
	radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
	print('\n Training set Accuracy for iteration' + str(i) + ' : ' + str(
		100 * np.mean((radialSvm_predicted_label == train_label).astype(float))) + '%')

	# Find the accuracy on Validation Dataset
	radialSvm_predicted_label = radialSvm.predict(validation_data)
	radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
	print('\n Validation set Accuracy for iteration' + str(i) + ' : ' + str(
		100 * np.mean((radialSvm_predicted_label == validation_label).astype(float))) + '%')

	# Find the accuracy on Testing Dataset
	radialSvm_predicted_label = radialSvm.predict(test_data)
	radialSvm_predicted_label = radialSvm_predicted_label.reshape((radialSvm_predicted_label.shape[0], 1))
	print('\n Testing set Accuracy for iteration' + str(i) + ' : ' + str(
		100 * np.mean((radialSvm_predicted_label == test_label).astype(float))) + '%')
"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
print('Multiclass')
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
