from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import utils
import tensorflow as tf
import matplotlib.pyplot as plt

train_test_split = 6000

FLAGS = None


def cnn(x):
  # reshape to use within a convolutional neural net.
  with tf.name_scope('reshape'):
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])

  # convolutional layer: maps one grayscale image to 32 feature maps
  with tf.name_scope('conv1'):
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
    h1 = tf.nn.relu(tf.nn.conv2d(x_reshaped, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)

  # pooling: downsamples by 2X
  with tf.name_scope('pool1'):
    h2 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # convolutional layer: maps 32 feature maps to 64
  with tf.name_scope('conv2'):
    W3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b3 = tf.Variable(tf.truncated_normal([64], stddev=0.1))
    h3 = tf.nn.relu(tf.nn.conv2d(h2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)

  # pooling layer: downsamples by 2x
  with tf.name_scope('pool2'):
    h4 = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # fully connected layer
  # after 2 round ofs downsampling 28x28 image = 7x7x64 features
  # maps to 1024 features
  with tf.name_scope('fc1'):
    W5 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    b5 = tf.Variable(tf.truncated_normal([1024], stddev=0.1))
    h5_flat = tf.reshape(h4, [-1, 7*7*64])
    h5 = tf.nn.relu(tf.matmul(h5_flat, W5) + b5)

  # dropout
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_drop = tf.nn.dropout(h5, keep_prob)

  # map the 1024 features to 10 classes
  with tf.name_scope('fc2'):
    W7 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b7 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    y_pred = tf.matmul(h_drop, W7) + b7
  return y_pred, keep_prob



def run_model(train_data, train_labels, test_data, test_labels):
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  y_pred, keep_prob = cnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  accuracies = list()
  iterations = list()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(0, 4):
      print('epoch', epoch)
      # regenerate batches in each epoch
      batched = utils.generate_batches(train_data, train_labels, batch_size = 50)
      i = 0
      for batch in batched:
        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy))
          accuracies.append(train_accuracy)
          iterations.append(epoch * len(batched) + i)
          print(iterations[-1])
        i += 1
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))
    print(accuracies)
    print(iterations)
    #plt.plot(accuracies, iterations)
    #plt.show()

def split_train_test():
  data, labels = utils.read_from_csv(one_hot = True, filename = './data/fashion-mnist_train.csv')
  train_data = data[:-train_test_split]
  train_labels = labels[:-train_test_split]
  test_data = data[-train_test_split:]
  test_labels = labels[-train_test_split:]

  run_model(train_data, train_labels, test_data, test_labels)

def compressed_test():
  train_data, train_labels = utils.read_from_csv(one_hot = True, filename = './data/fashion-mnist_train.csv')
  test_data, test_labels = utils.read_from_csv(one_hot = True, filename = './data/10.csv')

  run_model(train_data, train_labels, test_data[0:10000], test_labels[0:10000])

#split_train_test()
compressed_test()