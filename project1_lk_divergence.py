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

W2_en = tf.Variable(tf.truncated_normal([500, 150], stddev = 0.1, seed = 123)) # encoder step
b2_en = tf.Variable(tf.zeros([150]))
W2_de = tf.transpose(W2_en) # decoder step
b2_de = tf.Variable(tf.zeros([500]))

W3 = tf.Variable(tf.truncated_normal([150, 10], stddev=0.1, seed = 123))
b3 = tf.Variable(tf.zeros([10]))

h1 = tf.nn.sigmoid(tf.matmul(x, W1_en) + b1_en) # hidden layer
h2 = tf.nn.sigmoid(tf.matmul(h1, W2_en) + b2_en)
y = tf.matmul(h2, W3) + b3

beta = 5e-6
rho = tf.constant(0.05, shape = [1, 500])
rho0 = tf.constant(0.05, shape = [1, 150])
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
train_step3 = tf.train.GradientDescentOptimizer(0.5).minimize(loss3, var_list=[W3, b3])

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
