from csv import reader
from numpy import *

##### Preparing datasets #####

r = 600 # I am going to study the first r samples in both the training and test datasets.

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

# Convert string column to integer
def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column].strip())

# load and prepare data
filename1 = 'mnist_train.csv'
mnist_train = load_csv(filename1)
mnist_train = mnist_train[0:r]

# convert string class to integers
for i in range(0, len(mnist_train[0])):
	str_column_to_int(mnist_train, i)

# convert dataset from list to array
mnist_train = asarray(mnist_train)

filename2 = 'mnist_test.csv'
mnist_test = load_csv(filename2)
mnist_test = mnist_test[0:r]

# convert string class to floats
for i in range(0, len(mnist_test[0])):
	str_column_to_int(mnist_test, i)

# convert dataset from list to array
mnist_test = asarray(mnist_test)


