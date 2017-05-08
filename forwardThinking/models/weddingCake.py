from __future__ import division, print_function
import numpy as np
from time import time

import keras
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model, Sequential
from keras import regularizers

class WeddingCake(object):
    """ Keras based implementation of push-Forward Stacking., aka Wedding Cake """

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.transform_weights = []
        self.summary = {}
        self.summary['model_name'] = 'WeddingCake'
        self.summary['model_version'] = '1.0'


    def fit(self, x_train, y_train, x_test, y_test, activation='relu', 
            loss_func='categorical_crossentropy', optimizer='adam', 
            epochs=10, reg_type='l1', reg=0, dropout=False, verbose=True):
        """ Train the model. 
        Args:
          x_train
          y_train
          activation
          loss_func
          optimizer
          epochs
          reg_type
          reg
          dropout
          verbose
        """
        if verbose: print("Starting Training...\n")
        
        # Regularization
        if reg_type == 'l1':
            r = regularizers.l1(reg)
        elif reg_type == 'l2':
            r = regularizers.l2(reg)
        else:
            r = regularizers.l1(0) #set to 0 for no regularization

        # Store model training info
        self.summary['num_instances']= x_train.shape[0]
        self.summary['num_features'] = x_train.shape[1]
    
        # Inital Training
        input_dim = self.layer_dims[0]
        output_dim = self.layer_dims[-1]

        t0 = time() # start time

        if verbose: print("[Training Layer 0]")
        if verbose: print("Num Features: %d" % input_dim)
        init_model = Sequential()
        if dropout : init_model.add(Dropout(0.25, input_shape=(input_dim,)))
        init_model.add(Dense(output_dim, activation='sigmoid', input_shape=(input_dim,), 
            kernel_regularizer=r, name='init_dense'))
        init_model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
        init_model.fit(x_train, y_train, epochs=epochs, verbose=verbose, validation_data=(x_test, y_test))
        frozen_weights = init_model.get_layer('init_dense').get_weights()
        
        acc_hist = []
        loss_hist = []
        test_acc_hist = []
        test_loss_hist = []
        for i, layer_dim in enumerate(self.layer_dims[1:]):
            # Build and train layer
            if verbose: print("[Training Layer %s]" % str(i+1))
            if verbose: print("Num Features: %d" % input_dim)
            layer = self._build_layer_model(input_dim, layer_dim, output_dim, frozen_weights,
                    freeze=self.freeze, activation=activation, loss_func=loss_func, optimizer=optimizer, 
                    reg_type=reg_type, reg=reg, dropout=dropout)

            layer_hist = layer.fit(x_train,  y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=verbose)
            acc_hist += layer_hist.history['acc']
            loss_hist += layer_hist.history['loss']
            test_acc_hist += layer_hist.history['val_acc']
            test_loss_hist += layer_hist.history['val_loss']

            # Transform input data
            t_weights = layer.get_layer('transform_dense').get_weights()
            self.transform_weights.append(t_weights)
            W, b = t_weights
            new_data = relu(np.dot(x_train, W) + b)
            x_train = np.hstack((x_train, new_data))
            new_data = relu(np.dot(x_test, W) + b)
            x_test = np.hstack((x_test, new_data))

            # Freeze the learned layer
            frozen_weights = self._flatten_layer(layer)

            input_dim += layer_dim

        # store final learned weights
        self.weights = frozen_weights

        # Save training stats
        self.summary['training_time'] = time() - t0
        self.summary['accuracy'] = acc_hist
        self.summary['loss'] = loss_hist
        self.summary['val_accuracy'] = test_acc_hist
        self.summary['val_loss'] = test_loss_hist


    def predict(x_test):
        """ Use model to predict x_test """
        # Transform the data
        for layer_weights in self.transform_weights:
            W, b = layer_weights
            new_data = relu(np.dot(x_test, W) + b)
            x_test = np.hstack((x_test, new_data))

        # Classify
        W, b = self.weights
        output = np.dot(x_test, W) + b
        # TODO add sigmoid
        r
