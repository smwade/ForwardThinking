from __future__ import print_function, division

import numpy as np
import json
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.utils.np_utils import to_categorical


class Layer(object):
    """
    A single layer for forward thinking. Trains a indavidual layer
    so that it can transform input data
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        # Set Hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes


    def fit(self, x_train, y_train, x_test, y_test, epochs=25, reg_amnt=0, learning_rate=.01,
            batch_size=32, nonlinear='relu', loss_func='categorical_crossentropy', 
            weight_scale=.1, early_stopping=False, reg_type=1, verbose=True):
        """ Train the layer. """

        if reg_type == 1:
            layer_reg = regularizers.l1(reg_amnt)
        else:
            layer_reg = regularizers.l2(reg_amnt)

        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        num_classes = self.num_classes
        self.nonlinear = nonlinear

        model = Sequential()
        model.add(Dense(hidden_dim, activation=nonlinear, input_shape=(input_dim,)))
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=layer_reg))
        model.compile(loss=loss_func, optimizer=Adam(), metrics=['accuracy'])
        t0 = time.time()
        self.history = model.fit(x_train, y_train,
                batch_size=batch_size,
                validation_data=(x_test, y_test),
                epochs=epochs,
                verbose=verbose)
        t1 = time.time()
        
        self.model = model
        self.W, self.b = self.model.get_weights()[0:2]
        self.layer_stats = {}
        self.layer_stats['batch_size'] = batch_size
        self.layer_stats['train_time'] = t1 - t0
        self.layer_stats['acc'] = self.history.history['acc'][-1]
        self.layer_stats['loss'] = self.history.history['loss'][-1]
        self.layer_stats['acc_hist'] = self.history.history['acc']
        self.layer_stats['loss_hist'] = self.history.history['loss']
        self.layer_stats['test_acc_hist'] = self.history.history['val_acc']
        self.layer_stats['test_loss_hist'] = self.history.history['val_loss']

        
    def predict(self, x_test):
        """ Predict output given data (used for final layer in ForwardThinking)"""
        return self.model.predict(x_test, verbose=False)
        
    def transform_data(self, data):
        """ Transorm data to hidden dimension. """
        h = np.dot(data, self.W) + self.b
        if self.nonlinear == 'relu':
            tranformed = np.maximum(h, 0)
        elif self.nonlinear == 'sigmoid':
            raise NotImplemented("Sigmoid not implemented")
        elif self.nonlinear == 'tanh':
            raise NotImplemented("tanh not implemented")

        return tranformed




class ForwardThinking(object):
    """ A feed forward layerwise training of neural networks. """
    
    def __init__(self, layer_dims, stack_data=True):
        """
        Args:
          layer_dims : (list) list of layer sizes (ie. [784,300,200,10])
          stack_data : (bool) pass on old data to layers
        """
        self.model_version = '1.0'
        self.layers = []
        self.layer_dims = layer_dims
        self.num_classes = layer_dims[-1]
        self.stack_data = stack_data

        self.summary = {}
        self.summary['model_name'] = 'ForwardThinking'
        self.summary['model_version'] = '1.0'
        self.summary['layer_dims'] = layer_dims
        self.summary['stacked_data'] = stack_data

        input_dim = layer_dims[0]
        for l in layer_dims[1:]:
            new_layer = Layer(input_dim, l, num_classes=self.num_classes)
            if self.stack_data:
                input_dim += l
            else:
                input_dim = l
            self.layers.append(new_layer)


    def fit(self, x_train, y_train, x_test, y_test, epochs=25, reg_amnt=0, learning_rate=.01,
            batch_size=32, nonlinear='relu', loss_func='categorical_crossentropy',
            weight_scale=.1, early_stopping=False, reg_type=1, verbose=True):
        """ Train the forward thinking model.
        Args:
          x_train : (array)
          y_train : (array)
          epochs : (int)
          reg_amnt : (float)
          reg_type : (int) regularization type from [1,2]
          learning_rate : (float)
          batch_size : (int)
          nonlinear : (string)
          weight_scale : (float)
        """ 
	# store parameters
        self.num_instances = x_train.shape[0]
        self.num_features = x_train.shape[1]
	self.params = {}
	for k, v in locals().iteritems():
            if k == 'kwargs':
                for k_p, v_p in v.iteritems():
                    self.params[k_p] = v_p
            else:
                if k not in ['x_train', 'y_train', 'self', 'verbose']:
                    self.params[k] = v

        # one hot encode data if necissary
        if y_train.ndim == 1:
            y_train =  to_categorical(y_train) 

        if verbose: print("Starting Training for PassForwardThinking")
        t0 = time.time()
        for i, layer in enumerate(self.layers):
            if verbose: print("[Training Layer %s]" % i)
            layer.fit(x_train, y_train, x_test, y_test, epochs=epochs, reg_amnt=reg_amnt, learning_rate=learning_rate, 
                    batch_size=batch_size, nonlinear=nonlinear, loss_func=loss_func,
                      weight_scale=weight_scale, early_stopping=early_stopping, reg_type=reg_type, verbose=verbose)
            if self.stack_data:
                transformed_data = layer.transform_data(x_train)
                x_train = np.hstack((x_train, transformed_data))
                transformed_data = layer.transform_data(x_test)
                x_test = np.hstack((x_test, transformed_data))
            else:
                x_train = layer.transform_data(x_train)
                x_test = layer.transform_data(x_test)

        self.summary['training_time'] = time.time() - t0
        self.summary['num_instances'] = self.num_instances 
        self.summary['num_features'] = self.num_features
        self.summary['accuracy'] = self.layers[-1].layer_stats['acc']
        self.summary['loss'] = self.layers[-1].layer_stats['loss']
        self.summary['parameters'] = self.params

        acc_list = []
        test_acc_list = []
        for layer in self.layers:
            acc_list += layer.layer_stats['acc_hist']
            test_acc_list += layer.layer_stats['test_acc_hist']
        self.summary['accuracy'] = acc_list
        self.summary['val_accuracy'] = test_acc_list

        loss_hist = []
        test_loss_hist = []
        for layer in self.layers:
            loss_hist += layer.layer_stats['loss_hist']
            test_loss_hist += layer.layer_stats['test_loss_hist']
        self.summary['loss'] = loss_hist
        self.summary['val_loss'] = test_loss_hist
        

        if verbose: print("Trained model in %s seconds" % self.training_time)


    def predict(self, x_test):
        """ Predict with the trained model.
        Args:
          x_test : (array)
        """
        for i, layer in enumerate(self.layers[:-1]):
            if self.stack_data:
                transformed_data = layer.transform_data(x_test)
                x_test = np.hstack((x_test, transformed_data))
            else:
                x_test = layer.transform_data(x_test)
        final_layer = self.layers[-1]
        return np.argmax(final_layer.predict(x_test), axis=1)
