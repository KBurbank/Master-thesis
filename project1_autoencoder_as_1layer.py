import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import matplotlib.pyplot as plt
from pylab import *

import tensorflow as tf
import numpy as np
tf.set_random_seed(123)
np.random.seed(123)

x = tf.placeholder(tf.float32, [None, 784])  # Input layer
y_ = tf.placeholder(tf.float32, [None, 10])  # True values for output layer
W1e = tf.Variable(tf.truncated_normal([784, 100], stddev = 0.1, seed = 123)) # encoder step
b1e = tf.Variable(tf.zeros([100]))
b1d = tf.Variable(tf.zeros([784]))
W2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1, seed = 123)) # fully connected layer
b2 = tf.Variable(tf.zeros([10]))

h = tf.nn.sigmoid(tf.matmul(x, W1e) + b1e) # hidden layer
y = tf.matmul(h, W2) + b2

loss1 = tf.reduce_sum(tf.square(tf.nn.sigmoid(tf.matmul(h, tf.transpose(W1e))+b1d) - x))
loss2 = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step1 = tf.train.GradientDescentOptimizer(0.5).minimize(loss1, var_list=[W1e, b1e, b1d])
train_step2 = tf.train.GradientDescentOptimizer(0.5).minimize(loss2, var_list=[W2, b2])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train
for _ in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step1, feed_dict={x: batch_xs})

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step2.run(feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))

#visualize filters
for i in range(100):
    pixels = W1e[:,i].eval() # convert tensor to array
    pixels = pixels.reshape((28, 28))
    plt.subplot(10, 10, i+1)
    plt.imshow(pixels, cmap = 'winter')
    plt.axis('off')
    

