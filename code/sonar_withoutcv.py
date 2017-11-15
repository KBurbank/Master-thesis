from csv import reader
from numpy import *
import numpy as np

##### Preparing dataset #####

# Load a CSV file
def load_csv(filename):
	dataset = list()
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

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# load and prepare data
filename = 'sonar.all-data'
dataset = load_csv(filename)

# convert string class to floats
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)

# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)

# convert dataset from list to array
dataset = asarray(dataset)

### Algorithm ###

l_rate = 0.01        # Learning rate
n_epoch = 1000  		  # Number of iterations
l_hidden_layer = 61  # Number of neurons in hidden layer

# Training weights
np.random.seed(222)
# Mean zero; The reason why I devide the whole term by 100 is that the weights were too big for this dataset.
weights1 = (2*np.random.random((l_hidden_layer, len(dataset[0, :])-1)) -1)/100
bias1 = np.zeros((1, len(dataset[:,0])))
weights2 = (2*np.random.random((1, l_hidden_layer))-1)/100
bias2 = np.zeros((1, len(dataset[:,0])))

for epoch in range(n_epoch):
	hidden = np.dot(weights1, (dataset[:, 0:60]).T) + np.repeat(bias1, l_hidden_layer, axis = 0)
	# hidden represents the values in hidden layer.
	hidden = 1 / (1 + np.exp(-hidden))
	prediction = np.dot(weights2, hidden) + bias2
	#prediction = 1 / (1 + np.exp(-prediction))
	#  I am going to use step function here. In this way I can get the right value.
	if True:
		for i in range(len(prediction[0, :])):
			prediction[0, i] = 1.0 if prediction[0,i] >= 0.0 else 0.0
	error2 = dataset[:, -1] - prediction
	error1 = np.dot(weights2.T, error2)
	weights2 += l_rate * np.dot(error2, hidden.T)
	#weights2 *= 0.8
	bias2 += l_rate * error2
	weights1 += l_rate * np.dot(error1, dataset[:, 0:60])
	#weights1 *= 0.8
	bias1 += l_rate * error1[0, ]
print prediction

# Evaluating algorithm
error = 0
for j in range(len(prediction[0, :])):
	error += (dataset[j, -1] - prediction[0, j])*(dataset[j, -1] - prediction[0, j])
correct = 100 - error / float(len(prediction[0, :])) * 100.0

print correct