from __future__ import division, print_function

import numpy as np
import tensorflow as tf

class PushForwardThinking(object):
    """ An implementation of push forward thinking. """

    def __init__():
        pass

    def _create_palaceholders(self, n_features, n_classes):
        self.input_data = tf.placeholder(
                tf.float32, [None, n_features], name='x')
        self.input_labels = tf.placeholder(
                tf.float32, [None, n_classes], name='y')


