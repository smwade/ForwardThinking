from __future__ import division, print_function
import numpy as np
from time import time

import keras
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K

def prelu_init(shape, dtype=None):
    return -1

class WeddingCake(object):
    """ Keras based implementation of push-Forward Stacking., aka Wedding Cake """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.transform_weights = []
        self.summary = {}
        self.summary['model_name'] = 'WeddingCake'
        self.summary['model_version'] = '1.0'


    def _extend(self, model):
        """ Add a hidden layer to the model. """
        # Create model
        new_model = Sequential()
        new_model.add(Dense(self.hidden_dim, input_shape=(self.input_dim,), name="dense_0"))

        dense_idx = 1
        act_indx = 1
        for l in model.layers[1:-1]:
            if "dense" in l.name:
                new_model.add(Dense(l.output_shape[1], input_shape=(l.input_shape[1],), name="dense_{}".format(dense_idx)))
                dense_idx += 1
            if "activation" in l.name:
                new_model.add(PReLU(alpha_initializer=prelu_init, name='activation_{}'.format(act_indx)))
                act_indx += 1

        # new layer
        new_model.add(Dense(self.hidden_dim, activation='linear',
                            kernel_initializer = keras.initializers.Identity(gain=1.0), 
                            bias_initializer='zeros', name="dense_{}".format(dense_idx)))
        new_model.add(PReLU(alpha_initializer=prelu_init, name='activation_{}'.format(act_indx)))
        
        # same output layer
        old_weights = model.get_layer("output_dense").get_weights()
        new_model.add(Dense(self.output_dim, name="output_dense"))
        new_model.get_layer("output_dense").set_weights(old_weights)

        # Set the old weights
        dense_idx = 0
        act_indx = 1
        for l in model.layers[:-3]:
            if "dense" in l.name:
                old_weights = model.get_layer("dense_{}".format(dense_idx)).get_weights()
                new_model.get_layer('dense_{}'.format(dense_idx)).set_weights(old_weights)
                dense_idx += 1
            if "activation" in l.name:
                old_weights = model.get_layer("activation_{}".format(act_indx)).get_weights()
                new_model.get_layer('activation_{}'.format(act_indx)).set_weights(old_weights)
                act_indx += 1

        new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return new_model


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
        acc_hist = []
        loss_hist = []
        test_acc_hist = []
        test_loss_hist = []
    
        t0 = time() # start time

        if verbose: print("[Training Layer 0]")
        model = Sequential()
        if dropout : model.add(Dropout(0.25, input_shape=(self.input_dim,)))
        model.add(Dense(self.hidden_dim, activation='relu', input_shape=(self.input_dim,), 
            kernel_regularizer=r, name='dense_0'))
        model.add(Dense(self.output_dim, activation='sigmoid', kernel_regularizer=r, name='output_dense'))
        model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
        layer_hist = model.fit(x_train, y_train, epochs=epochs, verbose=verbose, validation_data=(x_test, y_test))
        model.summary()
        acc_hist += layer_hist.history['acc']
        loss_hist += layer_hist.history['loss']
        test_acc_hist += layer_hist.history['val_acc']
        test_loss_hist += layer_hist.history['val_loss']
        
        for i in range(self.num_layers-1):
            # Build and train layer
            if verbose: print("[Training Layer %s]" % str(i+1))
            model = self._extend(model) 

            layer_hist = model.fit(x_train,  y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=verbose)
            model.summary()
            acc_hist += layer_hist.history['acc']
            loss_hist += layer_hist.history['loss']
            test_acc_hist += layer_hist.history['val_acc']
            test_loss_hist += layer_hist.history['val_loss']

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
