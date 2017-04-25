""" Forward Thinking """
from __future__ import division, print_function
import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main(_):

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Placeholders
    y_true = tf.placeholder(tf.float32, [None, 10])
    x = tf.placeholder(tf.float32, [None, 784])

    # Layer 1
    W1 = weight_variable([784, 100])
    b1 = bias_variable([100])
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    # Layer 2
    W2 = weight_variable([100, 10])
    b2 = bias_variable([10])
    y = tf.matmul(h1, W2) + b2

    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_true))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Start session
    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())

    # Train
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        if i % 1000 == 0:
            _, acc = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_true: batch_ys})
            print("step [%d]: training accuracy %g" % (i, acc))
        else:
            sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})
            

    # Test trained model
    print("--- RESULTS ---")
    print("Test Accuracy: %g" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                y_true: mnist.test.labels}))
    sess.graph
    sess.close()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=25, help='The number of epochs')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
