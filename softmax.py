from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import utils

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

train_test_split = 6000

def train(test_data, test_labels):
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  batched = utils.generate_batches(train_data, train_labels, batch_size = 500)
  print('number of batches', len(batched))
  i = 0
  for batch in batched:
    if i % 10 == 0:
      print('running batch', i, '...')
    i += 1
    batch_xs = batch[0]
    batch_ys = batch[1]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  return sess, y, y_, x

def test(sess, y, y_, test_data, test_labels, x):
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: test_data, y_: test_labels}))

data, labels = utils.read_from_csv(one_hot = True, filename = './data/fashion-mnist_train.csv')
print('data shape', data.shape)
print('labels shape', labels.shape)
train_data = data[:-train_test_split]
train_labels = labels[:-train_test_split]
test_data = data[-train_test_split:]
test_labels = labels[-train_test_split:]
sess, y, y_, x = train(train_data, train_labels)
test(sess, y, y_, test_data, test_labels, x)
