import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
from pylab import *

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 784])  # Input layer
y_ = tf.placeholder(tf.float32, [None, 10])  # True values for output layer
W1_en = tf.Variable(tf.truncated_normal([784, 500], stddev = 0.1, seed = 123)) # encoder step
b1_en = tf.Variable(tf.zeros([500]))
W1_de = tf.transpose(W1_en) # decoder step
b1_de = tf.Variable(tf.zeros([784]))

W2_en = tf.Variable(tf.truncated_normal([500, 144], stddev = 0.1, seed = 123)) # encoder step
b2_en = tf.Variable(tf.zeros([144]))
W2_de = tf.transpose(W2_en) # decoder step
b2_de = tf.Variable(tf.zeros([500]))

#W3 = tf.Variable(tf.truncated_normal([144, 10], stddev=0.1, seed = 123))
#b3 = tf.Variable(tf.zeros([10]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')

h1 = tf.nn.sigmoid(tf.matmul(x, W1_en) + b1_en) # hidden layer
h2 = tf.nn.sigmoid(tf.matmul(h1, W2_en) + b2_en)
#y = tf.matmul(h2, W3) + b3

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_image = tf.reshape(h2, [-1, 12, 12, 1])

h_conv1 = tf.nn.relu(conv2d(h_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([3*3*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y = tf.matmul(h_fc1, W_fc2) + b_fc2

beta = 5e-6
rho = tf.constant(0.05, shape = [1, 500])
rho0 = tf.constant(0.05, shape = [1, 144])

#kl_div_loss1 = tf.contrib.distributions.kl(rho, tf.reduce_mean(h1,1))
kl_div_loss1 = tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels = rho, logits = rho/tf.reduce_mean(h1,0)))
loss1 = tf.reduce_mean(tf.square(tf.nn.sigmoid(tf.matmul(h1, W1_de)+b1_de) - x)) + beta * kl_div_loss1

#kl_div_loss2 = tf.contrib.distributions.kl(rho, tf.reduce_mean(h2,1))
kl_div_loss2 = tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels = rho0, logits = rho0/tf.reduce_mean(h2,0)))
loss2 = tf.reduce_mean(tf.square(tf.nn.sigmoid(tf.matmul(h2, W2_de)+b2_de) - h1))+ beta * kl_div_loss2

loss3 = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step1 = tf.train.GradientDescentOptimizer(10).minimize(loss1, var_list=[W1_en, b1_en, b1_de])
train_step2 = tf.train.GradientDescentOptimizer(10).minimize(loss2, var_list=[W2_en, b2_en, b2_de])
train_step3 = tf.train.GradientDescentOptimizer(1e-4).minimize(loss3, var_list=[W_conv1, b_conv1, W_conv2, b_conv2,
                                                                               W_fc1, b_fc1, W_fc2, b_fc2])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train
for _ in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step1, feed_dict={x: batch_xs})

for _ in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step2, feed_dict={x: batch_xs})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for _ in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    train_step3.run(feed_dict={x: batch_xs, y_: batch_ys})


# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))

# visualize filters
if False:
    pixels = W1_en[:,208].eval() # convert tensor to array
    pixels = pixels.reshape((28, 28))
    imshow(pixels, cmap = 'winter')
    show()