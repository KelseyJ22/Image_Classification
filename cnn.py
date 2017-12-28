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
  """cnn builds the graph for a deep net for classifying images
  Args:
    x: an input tensor with the dimensions (N_examples, 784)
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10),
    with values equal to the logits of classifying into one of 10 classes.
    keep_prob is a scalar placeholder for the probability of dropout.
  """
  # reshape to use within a convolutional neural net.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # convolutional layer: maps one grayscale image to 32 feature maps
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # pooling: downsamples by 2X
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # convolutional layer: maps 32 feature maps to 64
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # pooling layer: downsamples by 2x
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # fully connected layer
  # after 2 round of downsampling the 28x28 image = 7x7x64 feature maps
  # this maps to 1024 features
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # dropout
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # map the 1024 features to 10 classes
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  data, labels = utils.read_from_csv(one_hot = True, filename = './data/fashion-mnist_train.csv')
  print('data shape', data.shape)
  print('labels shape', labels.shape)
  train_data = data[:-train_test_split]
  train_labels = labels[:-train_test_split]
  test_data = data[-train_test_split:]
  test_labels = labels[-train_test_split:]

  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  y_conv, keep_prob = cnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
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
    score = 0
    epoch = 0
    while True:
      print('epoch', epoch)
      # regenerate batches in each epoch
      batched = utils.generate_batches(train_data, train_labels, batch_size = 50)
      i = 0
      for batch in batched:
        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy))
          accuracies.append(train_accuracy)
          iterations.append(epoch * 100 + i)
          if train_accuracy >= 0.96:
            break
        i += 1
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        epoch += 1

    print('test accuracy %g' % accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))
    print(accuracies)
    print(iterations)
    #plt.plot(accuracies, iterations)
    #plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='./data/fashion-mnist_train.csv', help='input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)