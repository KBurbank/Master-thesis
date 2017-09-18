import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import matplotlib.pyplot as plt

#pixels = mnist.train.images[15, :]
#pixels = pixels.reshape((28, 28))
#plt.imshow(pixels, cmap = 'winter')
#plt.axis('off')

import tensorflow as tf
import numpy as np

tf.set_random_seed(123)
np.random.seed(123)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1, seed = 123)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')

num_filter1 = 32
num_filter2 = 64

x = tf.placeholder(tf.float32, [None, 784])  # Input layer
y_ = tf.placeholder(tf.float32, [None, 10])  # True values for output layer

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, num_filter1])
b_conv1 = bias_variable([num_filter1])
b_conv1_de = bias_variable([1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

x_image_recon = tf.nn.relu(conv2d(h_conv1, tf.transpose(W_conv1, perm=[1, 0, 3, 2])) + b_conv1_de)

W_conv2 = weight_variable([3, 3, num_filter1, num_filter2])
b_conv2 = bias_variable([num_filter2])
b_conv2_de = bias_variable([num_filter1])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool1_recon= tf.nn.relu(conv2d(h_conv2, tf.transpose(W_conv2, perm=[1, 0, 3, 2]))+b_conv2_de)

W_fc1 = weight_variable([7*7*num_filter2, 10])
b_fc1 = bias_variable([10])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filter2])
y = tf.matmul(h_pool2_flat, W_fc1)+b_fc1

loss1 = tf.reduce_mean(tf.square(x_image_recon - x_image))
loss2 = tf.reduce_mean(tf.square(h_pool1_recon - h_pool1))
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step1 = tf.train.GradientDescentOptimizer(0.5).minimize(loss1, var_list=[W_conv1, b_conv1, b_conv1_de])
train_step2 = tf.train.GradientDescentOptimizer(0.5).minimize(loss2, var_list = [W_conv2, b_conv2, b_conv2_de])
train_step3 = tf.train.GradientDescentOptimizer(0.5).minimize(loss3, var_list = [W_fc1, b_fc1])
#train_step4 = tf.train.GradientDescentOptimizer(1e-2).minimize(loss3) # Need to be changed to be safe for not doing
# initialization again

#parellel network
W_conv1_prime = W_conv1
b_conv1_prime = b_conv1
b_conv1_de_prime = b_conv1_de

h_conv1_prime = tf.nn.relu(conv2d(x_image, W_conv1_prime) + b_conv1_prime)
h_pool1_prime = max_pool_2x2(h_conv1_prime)

W_conv2_prime = W_conv2
b_conv2_prime = b_conv2
b_conv2_de_prime = b_conv2_de

h_conv2_prime = tf.nn.relu(conv2d(h_pool1_prime, W_conv2_prime)+b_conv2_prime)
h_pool2_prime = max_pool_2x2(h_conv2_prime)

W_fc1_prime = W_fc1
b_fc1_prime = b_fc1

h_pool2_prime_flat = tf.reshape(h_pool2_prime, [-1, 7*7*num_filter2])
y_prime = tf.matmul(h_pool2_prime_flat, W_fc1_prime)+b_fc1_prime

loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_prime))

train_step4 = tf.train.GradientDescentOptimizer(1e-2).minimize(loss4, var_list = [W_conv1_prime, b_conv1_prime,
                                                                                   W_conv2_prime, b_conv2_prime,
                                                                                   W_fc1_prime, b_fc1_prime])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train
for _ in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step1, feed_dict={x: batch_xs})

for _ in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step2, feed_dict={x: batch_xs})

for _ in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step3, feed_dict={x: batch_xs, y_:batch_ys})

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step4, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images[0:1000, :],
                                    y_: mnist.test.labels[0:1000, :]}))