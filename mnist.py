from csv import reader
import numpy as np
from pylab import *

                                ##### Preparing datasets #####
r = 1800 # I am going to study the first r samples in both the training and test datasets.

                            ### Functions for loading dataset ###
# Load a CSV file
def load_csv(filename):
	dataset = []
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

					 ### Getting training and test dataset ###
# load and prepare training data
filename1 = 'mnist_train.csv'
mnist_train = load_csv(filename1)
mnist_train = mnist_train[0:r]
# convert string class to integers
for i in range(0, len(mnist_train[0])):
	str_column_to_float(mnist_train, i)
# convert dataset from list to array
mnist_train = asarray(mnist_train)
# centering the data (0~255 to -1~1)
mnist_train[:, 1:785] = (mnist_train[:, 1:785]-128)/128

# load and prepare test data
filename2 = 'mnist_test.csv'
mnist_test = load_csv(filename2)
mnist_test = mnist_test[0:r]
# convert string class to integers
for i in range(0, len(mnist_test[0])):
	str_column_to_float(mnist_test, i)
# convert dataset from list to array
mnist_test = asarray(mnist_test)
# centering the data (0~255 to -1~1)
mnist_test[:, 1:785] = (mnist_test[:, 1:785]-128)/128

# show one image
if False:
	pixels = mnist_train[15, 1:len(mnist_train[0,:])]
	pixels = pixels.reshape((28, 28))
	imshow(pixels, cmap = 'spring')
	show()

# convert labels into one-hot vectors
label_train = mnist_train[:, 0].astype(int)
label_train_onehot = zeros((len(mnist_train[:, 0]), 10))
label_train_onehot[range(len(mnist_train[:, 0])), label_train] = 1
label_test = mnist_test[:, 0].astype(int)
label_test_onehot = zeros((len(mnist_test[:, 0]), 10))
label_test_onehot[range(len(mnist_test[:, 0])), label_test] = 1

									##### Algorithm #####
l_rate = 0.0001        # Learning rate
n_epoch = 200 		  # Number of iterations
l_hidden_layer = 300  # Number of neurons in hidden layer

                                  ### Training weights ###
#def softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    return exp(x) / sum(exp(x), axis=0)
np.random.seed(222)
# weights and biases initialization
weights1 = 2*np.random.random((l_hidden_layer, len(mnist_train[0, :])-1)) -1
bias1 = zeros((l_hidden_layer, 1))
weights2 = 2*np.random.random((10, l_hidden_layer))-1
bias2 = zeros((10, 1))
n_batch = 600 # Size for each minibatch

for epoch in range(n_epoch):
	for i in range(0, r, n_batch):
		mnist_train_i = mnist_train[i:i+n_batch, :]
		label_train_onehot_i = label_train_onehot[i:i+n_batch, :]
		hidden = np.dot(weights1, (mnist_train_i[:, 1:len(mnist_train_i[0, :])]).T) + np.repeat(bias1, len(mnist_train_i[:, 1]), axis = 1)
		# hidden represents the values in hidden layer.
		hidden = 1 / (1 + np.exp(-hidden))
		prediction = np.dot(weights2, hidden) + np.repeat(bias2, len(mnist_train_i[:, 1]), axis = 1)
		# prediction represents the input values which go into the output layer
		#for i in range(len(prediction[0, :])):
		#	prediction[:, i] = softmax(prediction[:, i])
		prediction = argmax(prediction, axis = 0)
		prediction_onehot = zeros((10, len(prediction)))
		prediction_onehot[prediction, range(len(prediction))] = 1
		error2 = (label_train_onehot_i).T - prediction_onehot
		error1 = np.dot(weights2.T, error2)
		weights2 += l_rate * np.dot(error2, hidden.T)
		bias2 = (mean(error2, axis=1) + bias2.T).T
		weights1 += l_rate * np.dot(error1, mnist_train_i[:, 1:len(mnist_train_i[0, :])])
		bias1 = (mean(error1, axis=1) + bias1.T).T
print prediction_onehot

								##### Evaluation #####
# Training error
train_error = (sum(abs(prediction_onehot - label_train_onehot_i.T)))/2
train_error_ratio = train_error/n_batch
print train_error_ratio
#Test error
hidden2 = np.dot(weights1, (mnist_test[:, 1:len(mnist_test[0, :])]).T) + bias1
hidden2 = 1 / (1 + np.exp(-hidden2))
prediction2 = np.dot(weights2, hidden2) + bias2
#for i in range(len(prediction2[0, :])):
#	prediction2[:, i] = softmax(prediction2[:, i])
prediction2 = argmax(prediction2, axis = 0)
prediction_onehot2 = zeros((10, len(prediction2)))
prediction_onehot2[prediction2, range(len(prediction2))] = 1
test_error = sum(abs(prediction_onehot2 - label_test_onehot.T))/2
test_error_ratio = test_error/r
print test_error_ratio

# For fun
pixels = weights1[15,:]
pixels = pixels.reshape((28, 28))
imshow(pixels, cmap = 'winter')
show()
# No specific pattern is shown 
print 'thank you'
