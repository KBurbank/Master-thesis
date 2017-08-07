import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
from pylab import *

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

x = tf.placeholder(tf.float32, [None, 784])  # Input layer
y_ = tf.placeholder(tf.float32, [None, 10])  # True values for output layer
W1_en = tf.Variable(tf.truncated_normal([784, 500], stddev = 0.1, seed = 123)) # encoder step
b1_en = tf.Variable(tf.zeros([500]))
W1_de = tf.Variable(tf.truncated_normal([500, 784], stddev = 0.1, seed = 123)) # decoder step
b1_de = tf.Variable(tf.zeros([784]))

W2_en = tf.Variable(tf.truncated_normal([500, 150], stddev = 0.1, seed = 123)) # encoder step
b2_en = tf.Variable(tf.zeros([150]))
W2_de = tf.Variable(tf.truncated_normal([150, 500], stddev = 0.1, seed = 123)) # decoder step
b2_de = tf.Variable(tf.zeros([500]))

W3 = tf.Variable(tf.truncated_normal([150, 10], stddev=0.1, seed = 123))
b3 = tf.Variable(tf.zeros([10]))

h1 = tf.nn.sigmoid(tf.matmul(x, W1_en) + b1_en) # hidden layer
h2 = tf.nn.sigmoid(tf.matmul(h1, W2_en) + b2_en)
y = tf.matmul(h2, W3) + b3

loss1 = tf.reduce_mean(tf.square(tf.nn.sigmoid(tf.matmul(h1, W1_de)+b1_de) - x))
loss2 = tf.reduce_mean(tf.square(tf.nn.sigmoid(tf.matmul(h2, W2_de)+b2_de) - h1))
loss3 = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step1 = tf.train.GradientDescentOptimizer(10).minimize(loss1, var_list=[W1_en, b1_en, W1_de, b1_de])
train_step2 = tf.train.GradientDescentOptimizer(0.05).minimize(loss2, var_list=[W2_en, b2_en, W2_de, b2_de])
train_step3 = tf.train.GradientDescentOptimizer(0.5).minimize(loss3, var_list=[W3, b3])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train

loss1_data = np.zeros([1000, 1], dtype= float)
loss2_data = np.zeros([1000, 1], dtype= float)

for i in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step1, feed_dict={x: batch_xs})
    loss1_data[i, 0] = np.asarray(sess.run([loss1], feed_dict={x:batch_xs}))

plt.plot(loss1_data)
plt.show()

for i in range(1000):
    batch_xs, batch_ys= mnist.train.next_batch(100)
    sess.run(train_step2, feed_dict={x: batch_xs})
    loss2_data[i, 0] = np.asarray(sess.run([loss2], feed_dict={x: batch_xs}))

plt.plot(loss2_data)
plt.show()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
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
