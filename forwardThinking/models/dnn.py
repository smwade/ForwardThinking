'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.utils.np_utils import to_categorical



class DNN(object):

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims


    def fit(self, x_train, y_train, nonlinear='relu', loss_func='categorical_crossentropy', learning_rate=.01, 
            reg_type=2, reg=0, epochs=20, batch_size=32, verbose=True):
        """ 
        Train the DNN

        Args:
          x_train : (array)
          y_train : (array)
          nonlinear : (string)
          loss_func : (string) ['sse', 'cross-entropy']
          learning_rate : (float)
          reg_type : (string)
          reg : (float)
          epochs : (int)
          batch_size : (int)
          weights_hist : (bool)
        """
        if y_train.ndim == 1:
            y_train = to_categorical(y_train)

        if reg_type == 1:
            layer_reg = regularizers.l1(reg)
        else:
            layer_reg = regularizers.l2(reg)

        layer_dims = self.layer_dims

        model = Sequential()
        for l in range(len(layer_dims)-2):
            model.add(Dense(layer_dims[l+1], activation=nonlinear, 
                input_shape=(layer_dims[l],), kernel_regularizer=layer_reg))

        model.add(Dense(layer_dims[-1], activation='softmax'))
        model.compile(loss=loss_func, optimizer=Adam(), metrics=['accuracy'])
        self.history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose)
        self.model = model


    def predict(self, x_test):
        return self.model.predict_classes(x_test)
