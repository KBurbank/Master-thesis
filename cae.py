import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

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

x = tf.placeholder(tf.float32, [None, 784])  # Input layer
y_ = tf.placeholder(tf.float32, [None, 10])  # True values for output layer

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([4, 3, 1, 32])
b_conv1 = bias_variable([32])
b_conv1_de = bias_variable([1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

x_image_recon = tf.nn.relu(conv2d(h_conv1, tf.transpose(W_conv1, perm=[1, 0, 3, 2])) + b_conv1_de)

W_fc1 = weight_variable([14*14*32, 10])
b_fc1 = bias_variable([10])

h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
y = tf.nn.sigmoid(tf.matmul(h_pool1_flat, W_fc1)+b_fc1)


loss1 = tf.reduce_mean(tf.square(x_image_recon - x_image))
loss2 = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step1 = tf.train.GradientDescentOptimizer(0.5).minimize(loss1, var_list=[W_conv1, b_conv1, b_conv1_de])
train_step2 = tf.train.GradientDescentOptimizer(0.5).minimize(loss2, var_list=[W_fc1, b_fc1])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train
for _ in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step1, feed_dict={x: batch_xs})

for _ in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step2, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
