import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
#Added by Abinash for randomly selecting validation training data and labels
from random import randrange

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    print("initializing weights")
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    print("initializing weights end")
    return W



def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1/(1+np.exp(-z))



def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    print("Preprocess starts")
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary

    #Pick a reasonable size for validation data
    # print(mat)
    # print(type(mat))
    # print(type(mat['train0']))
    #
    # print("Train 0 size")
    # print(len(mat['train0']))
    # print("Train 1 size")
    # print(len(mat['train1']))
    # print("Test 0 size")
    # print(len(mat['test0']))
    # print("Test 1 size")
    # print(len(mat['test1']))

    # print("Test total size")
    # print(len(mat['test0'])+len(mat['test1'])+len(mat['test2'])+len(mat['test3'])+len(mat['test4'])+len(mat['test5'])+len(mat['test6'])+len(mat['test7'])+len(mat['test8'])+len(mat['test9']))
    #print(mat['label0'])
    #print(vars(mat))
    # print("Train7 row 0 contents")
    # print(mat['train7'][0])
    #print("Train7 row 0 contents - after normalizing")
    #print(noramlizeRow(mat['train7'][0]))

    #initializing the lists
    trainingList = []
    testDataList = []
    validationDataList = []
    trainingLabelList = []
    testLabelList = []
    validationLabelList = []

    #validationListCounter = 0
    for key in mat:
        #print("key: %s , value: %s" % (key, mat[key]))
        #print("key: %s" % (key))
        if key[:5] == 'train' :
            for trainRow in mat[key]:
                #trainingList.append(trainRow) - mistake -a huge 1-d array
                #if validationListCounter:
                trainingList.append(noramlizeRow(trainRow))
                #extracting the label
                trainingLabelList.append(getLabelArray(key[5:]))
                #validationListCounter += 1
        elif key[:4] == 'test':
            for testRow in mat[key]:
                #trainingList.append(trainRow) - mistake -a huge 1-d array
                testDataList.append(noramlizeRow(testRow))
                #extracting the label
                testLabelList.append(getLabelArray(key[4:]))

    # print("Final training list total size")
    # print(len(trainingList))
    # print("Final training list row 0 contents")
    # print(trainingList[0])
    #
    # print("Final test data list total size")
    # print(len(testDataList))

    #this will store the indexes of the inut data part of the validaton data
    randomIndexHolder = {}

    #randomly seleting 10k validation data and labels
    while len(randomIndexHolder) < 10000:
        randomIndex = randrange(0,60000,1)
        if(randomIndex not in randomIndexHolder):
            validationDataList.append(trainingList[randomIndex])
            #setting the validation list labels
            validationLabelList.append(trainingLabelList[randomIndex])
            randomIndexHolder[randomIndex] = randomIndex

    newTrainingData = []
    newTrainingLabel = []
    for trainIndex in range(len(trainingList)):
        if(trainIndex not in randomIndexHolder):
            newTrainingData.append(trainingList[trainIndex])
            newTrainingLabel.append(trainingLabelList[trainIndex])

    # print("validation data list total size")
    # print(len(validationDataList))
    # print("Final validation data list row 0 contents")
    # print(validationDataList[0])
    # #vars(mat)
    #
    # print("Final training labels list total size")
    # print(len(trainingLabelList))
    # print("Final training label list row 0 contents")
    # print(trainingLabelList[0])

    #print("Final training label list first row for train1 contents")
    #print(trainingLabelList[len(mat['train0'])])

    # print("Final test labels list total size")
    # print(len(testLabelList))
    # print("Final test label list row 0 contents")
    # print(testLabelList[0])
    #
    # print("Final validation labels list total size")
    # print(len(validationLabelList))
    # print("Final validation label list row 0 contents")
    # print(validationLabelList[0])

    #TODO - trainingList has 60k rows and only 50k need to be assigned to train_data
    train_data = np.array(newTrainingData)
    train_label = np.array(newTrainingLabel)
    validation_data = np.array(validationDataList)
    validation_label = np.array(validationLabelList)
    test_data = np.array(testDataList)
    test_label = np.array(testLabelList)

    print("Preprocess ends")
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def getLabelArray(number):
    """ Added by Abinash
    This method is used to create the label array for the corresponding number
    The label array in this case will be 60kX10 vector - this method will be
    called 60k times i.e. by each training example and 10k times by the test
    examples returning a vector of size 10 each time."""
    labelHolder = {
        '0' : [1,0,0,0,0,0,0,0,0,0],
        '1' : [0,1,0,0,0,0,0,0,0,0],
        '2' : [0,0,1,0,0,0,0,0,0,0],
        '3' : [0,0,0,1,0,0,0,0,0,0],
        '4' : [0,0,0,0,1,0,0,0,0,0],
        '5' : [0,0,0,0,0,1,0,0,0,0],
        '6' : [0,0,0,0,0,0,1,0,0,0],
        '7' : [0,0,0,0,0,0,0,1,0,0],
        '8' : [0,0,0,0,0,0,0,0,1,0],
        '9' : [0,0,0,0,0,0,0,0,0,1]
    }
    #the second parameter is the default value to be returned
    return labelHolder.get(number, [])

def noramlizeRow(oneRowVector):
    """ Added by Abinash
    This will normalize each element of the vetor.
    Here we assume that the input is a 1-D vector"""
    #using broadcasting
    #Solved todo - check the below line for normalization - it leads to values more than 1
    # checked the below line and found that integer overflow was occurring on
    #squaring the elements - in order to convert to decimals dividing by 1.0
    oneRowVector = oneRowVector / 1.0
    #magnitude = np.sqrt(np.sum(np.power(oneRowVector,2)))
    #print(magnitude)
    magnitude = 255 #- for testing
    oneRowVector = oneRowVector / magnitude
    return oneRowVector


def layer(inputarr,weights):
    """% By Ranjan and Abinash - Takes the input values computes
	 the input to the next layer and applies the sigmoid function
    """
    intermediate=[]
    #k=0
    for i in range (0,len(weights)):
        #intermediate[k]=0
        k = 0
        for j in range (0,len(inputarr)):
            k += weights[i][j]*inputarr[j]
        intermediate.append(sigmoid(k))
        #intermediate[k]=sigmoid(intermediate[k])
        #k+=1
    return intermediate

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    print("nnobj starts")
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    #print(w1)
    #feedFwdTotalError = []
    #remove This
    # training_data = training_data[:15]
    # training_label = training_label[:15]
    training_data = np.array(training_data)
    training_data.resize(len(training_data),785)
    training_data[:, 784] = 1
    #w1 = np.array(w1)
    #w1.resize(50,785)
    ffOutputOfFirstLayer = np.array(sigmoid(np.dot(training_data,np.transpose(w1))))
    ffOutputOfFirstLayer.resize(len(ffOutputOfFirstLayer),n_hidden+1)
    ffOutputOfFirstLayer[:,n_hidden] = 1
    ffOutputOfSecondLayer = np.array(sigmoid(np.dot(ffOutputOfFirstLayer,np.transpose(w2))))
    SLEF = np.sum(np.square(np.subtract(ffOutputOfSecondLayer,training_label)),axis=1)*0.5
    feedFwdTotalErrorValue = np.sum(SLEF)/len(training_data)
    print("Feedforward & Error Calculation done")
    #calculating the max value from the feedfwdOutput vector of size 60kX10
    # here feedfwdOutput is the output of the second layer - ffOutputOfSecondLayer
    # for j in feedfwdOutput:
    #     ffmaxIndex = 0
    #     for k in range(len(j)):
    #         if j[k] > j[ffmaxIndex]:
    #             ffmaxIndex = k
    #     ffmaxOutput.append(ffmaxIndex)

    #comparing with the true label for calucalting the error
    #feedfwdError = np.subtract(feedfwdOutput,training_label)

    #squaring the error
    #feedfwdError = np.square(feedfwdError)

    #calucalting total error
    # feefwdTotalError = []
    #
    # for j in feedfwdError:
    #     feefwdTotalError.append(np.sum(j))

    #multiplying with 0.5 and dividing with the size of the training data
    #feefwdTotalErrorValue = np.sum(feedFwdTotalError) * 0.5 / len(training_data)
    # print("Check twice")
    # print("The total ffd error is:"+str(feefwdTotalErrorValue))

    #Delta calcualtionprint("ffOutputOfSecondLayer before delta calc -"+ str(ffOutputOfSecondLayer))
    # print("delta calcualtion **************" )
    # print("ffOutputOfSecondLayer before delta calc -"+ str(ffOutputOfSecondLayer))

    #print(np.ones(len(ffOutputOfSecondLayer),len(ffOutputOfSecondLayer[0])))
    # print("training_label output size :"+ str(len(training_label)) + " X " + str(len(training_label[0])) )
    # print("ffOutputOfSecondLayer output size :"+ str(len(ffOutputOfSecondLayer)) + " X " + str(len(ffOutputOfSecondLayer[0])) )

    #reaplced the dot products with multiply
    delta  = np.multiply(np.multiply(np.subtract(training_label,ffOutputOfSecondLayer) , np.subtract(1 , ffOutputOfSecondLayer) ), ffOutputOfSecondLayer )
    print("Delta done "+str(len(delta))+","+str(len(delta[0]))+" - "+str(len(ffOutputOfFirstLayer))+","+str(len(ffOutputOfFirstLayer[0])) )

    #multiplying delta with first layer output
    # print("first layer output size :"+ str(len(ffOutputOfFirstLayer)) + " X " + str(len(ffOutputOfFirstLayer[0])) )

    #derivative of the error function for W2 =   iterates through 1 to k and z iterates through 1 to m+1 (hidden units) -- do this for all input
    errorDerivative2 = np.multiply(np.dot(np.transpose(delta),ffOutputOfFirstLayer),-1)
#            for ffOutputOfFirstLayerElement in range(len(ffOutputOfFirstLayer[deltaRow])):
#                errorDerivative2[deltaRow][deltaElement][ffOutputOfFirstLayerElement] = delta[deltaRow][deltaElement]*ffOutputOfFirstLayer[deltaRow][ffOutputOfFirstLayerElement]
    print("Error Derivative 2 done")
    #errorDerivative2 = np.multiply(np.dot(delta,ffOutputOfFirstLayer), -1)
    # print(errorDerivative2)

    #we have to consider the biase nodes so the size of w1 is 50X785 and w2 is 10X51
    # print("Size of w1 "+str(len(w1))+" X "+ str(len(w1[0])))
    print("Size of w2 "+str(len(w2))+" X "+ str(len(w2[0])))

    #derivative of the error function for W2  - this is equivaent to summation of products
    #errorDerivative1 = np.dot(np.transpose(np.multiply(ffOutputOfFirstLayer*np.subtract(ffOutputOfFirstLayer,1),np.dot(np.transpose(np.dot(delta,w2[:50])),training_data))),training_data)
    # print(str(len(w2))+","+str(len(w2[0])))
    #ffOutputOfFirstLayer=ffOutputOfFirstLayer[:,:n_hidden]
    errorDerivative1 = np.dot(np.transpose(np.multiply(np.multiply(ffOutputOfFirstLayer,np.subtract(ffOutputOfFirstLayer,1)),np.dot(delta,w2))),training_data)
    print(str(len(errorDerivative1))+","+str(len(errorDerivative1[0])))
    #errorDerivative1Part1 = np.zeros((len(delta), len(w2[0])-1))
    #for deltaRow in range(len(delta)): # this is input - 50k
    #    for w2Element in range(len(w2[0])-1): # this is j
    #        for deltaElement in range(len(delta[deltaRow])): # this is l
    #            errorDerivative1Part1[deltaRow][w2Element] += delta[deltaRow][deltaElement]*w2[deltaElement][w2Element]
            #(multiply by   -- (z-1)z and updating the part1 of error derivative 1
    #        errorDerivative1Part1[deltaRow][w2Element] = ffOutputOfFirstLayer[deltaRow][w2Element]*np.subtract(ffOutputOfFirstLayer[deltaRow][w2Element],1)*errorDerivative1Part1[deltaRow][w2Element]
    #        for attributes in range(len(training_data[0])):
    #            errorDerivative1[deltaRow][w2Element][attributes] = errorDerivative1Part1[deltaRow][w2Element] * training_data[deltaRow][attributes]
    print("Error Derivative 1 done")

    # print(errorDerivative1)
    # print("errorDerivative1")
    # print(errorDerivative1[0][0])
    #
    # print("Size of errorDerivative1 "+str(len(errorDerivative1))+" X "+ str(len(errorDerivative1[0])))
    # print("Size of errorDerivative2 "+str(len(errorDerivative2))+" X "+ str(len(errorDerivative2[0])))

    #regularization
    obj_val = feedFwdTotalErrorValue + ((np.sum(np.square(w1)) + np.sum(np.square(w2))) * lambdaval * 0.5 / len(training_data))
    # print(obj_val)

    #calcualting the gradients
    grad_w1 = []
    grad_w2 = []
    grad_w2 = errorDerivative2 + w2*lambdaval
    errorDerivative1.resize(n_hidden,785)
    grad_w1 = errorDerivative1 + w1*lambdaval
    #grad_w1 = errorDerivative1 + w1*lambdaval

    grad_w1 = np.divide(grad_w1, len(training_data))
    grad_w2 = np.divide(grad_w2, len(training_data))
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    print(len(grad_w1))
    print(len(grad_w2))
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.concatenate((grad_w1, grad_w2),0)
    #obj_grad = np.array([])

    #for i in range(60):
    #    obj_grad = np.append(obj_grad,0)

    # print(obj_grad)
    print("nnobj ends")
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    print("Predict start")

    data = np.array(data)
    data.resize(len(data),785)
    data[:,784] = 1
    ffOutputOfFirstLayer = np.array(sigmoid(np.dot(data,np.transpose(w1))))
    ffOutputOfFirstLayer.resize(len(ffOutputOfFirstLayer),len(ffOutputOfFirstLayer[0])+1)
    ffOutputOfFirstLayer[:,len(ffOutputOfFirstLayer[0])-1] = 1
    ffOutputOfSecondLayer = np.array(sigmoid(np.dot(ffOutputOfFirstLayer,np.transpose(w2))))
    # data = data[:15]
    # doing a feed forward using the updated weights
    #ffOutputOfFirstLayer = []
    labels = []
    for output in ffOutputOfSecondLayer:
        #finding the max index of the second layer output
        maxVlaueIndex = np.argmax(output)
        #print(maxVlaue)
        #labels.append(getLabelArray(str(maxVlaueIndex)))
        labels.append(maxVlaueIndex)

    labels = np.array(labels)
    print("Predict end")
    #print(labels)
    return labels

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];
# .shape - returns a tuple of the dimensions

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.
print(" Initial weights")
print(len(initialWeights))
#w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
#w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
obj_val = 0
#print("LOOOOOL")
#print(w1)
#print("LOOOOL AGAIN")
#print(layer([1,2],[[1 , 2],[3 , 4]]))
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
#print("w1 and w2")
#print(w1)
#print(w2)
#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset
# train_label = train_label[:15]
# validation_label = validation_label[:15]
# test_label = test_label[:15]
newlabels = []
for output in train_label:
    #finding the max index of the second layer output
    maxVlaueIndex = np.argmax(output)
    #print(maxVlaue)
    #labels.append(getLabelArray(str(maxVlaueIndex)))
    newlabels.append(maxVlaueIndex)
#print("predicted label dimensions " + str(len(predicted_label)) + " X " + str(len(predicted_label[06])))
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == newlabels).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
newlabels = []
for output in validation_label:
    #finding the max index of the second layer output
    maxVlaueIndex = np.argmax(output)
    #print(maxVlaue)
    #labels.append(getLabelArray(str(maxVlaueIndex)))
    newlabels.append(maxVlaueIndex)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == newlabels).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
newlabels = []
for output in test_label:
    #finding the max index of the second layer output
    maxVlaueIndex = np.argmax(output)
    #print(maxVlaue)
    #labels.append(getLabelArray(str(maxVlaueIndex)))
    newlabels.append(maxVlaueIndex)
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == newlabels).astype(float))) + '%')
