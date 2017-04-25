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


    def fit(self, x_train, y_train, epochs=25, reg_amnt=0, learning_rate=.01,
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
                epochs=epochs,
                verbose=verbose)#validation_data=(x_test, y_test))
        t1 = time.time()
        
        self.model = model
        self.W, self.b = self.model.get_weights()[0:2]
        self.layer_stats = {}
        self.layer_stats['batch_size'] = batch_size
        self.layer_stats['train_time'] = t1 - t0
        self.layer_stats['acc'] = self.history.history['acc'][-1]
        self.layer_stats['loss'] = self.history.history['loss'][-1]

        
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
    """ A feed forward data transormation approach. """
    
    def __init__(self, layer_dims):
        """
        Args:
          layer_dims : (list)
        """
        self.layers = []
        input_dim = layer_dims[0]
        self.num_classes = layer_dims[-1]

        for l in layer_dims[1:]:
            new_layer = Layer(input_dim, l, num_classes=self.num_classes)
            input_dim = l
            self.layers.append(new_layer)

            
    def fit(self, x_train, y_train, epochs=25, reg_amnt=0, learning_rate=.01,
            batch_size=32, nonlinear='relu', loss_func='categorical_crossentropy',
            weight_scale=.1, early_stopping=False, reg_type=1, verbose=True):
        """ Fit the forward thinking model.
        Args:
          x_train : (array)
          y_train : (array)
          epochs : (int)
          reg_amnt : (float)
          learning_rate : (float)
          batch_size : (int)
          nonlinear : (string)
          weight_scale : (float)
        """  
        if y_train.ndim == 1:
            y_train =  to_categorical(y_train) 
        print("Starting Training for ForwardThinking")
        for i, layer in enumerate(self.layers):
            print("[ Training Layer %s ]" % i)
            layer.fit(x_train, y_train, epochs=epochs, reg_amnt=reg_amnt, learning_rate=learning_rate, 
                    batch_size=batch_size, nonlinear=nonlinear, loss_func=loss_func,
                      weight_scale=weight_scale, early_stopping=early_stopping, reg_type=reg_type, verbose=verbose)
            x_train = layer.transform_data(x_train)

            
    def predict(self, x_test):
        """ Predict with the trained model.
        Args:
          x_test : (array)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x_test = layer.transform_data(x_test)
        final_layer = self.layers[-1]
        return np.argmax(final_layer.predict(x_test), axis=1)
    
    
    def model_summary(self):
        print("Model Archetecture")
        print("-"*30)
        for i, layer in enumerate(self.layers):
            print("Layer: {}".format(i))
            print("Hidden: {}".format(layer.hidden_dim))
            print("Time: {:.4}".format(layer.layer_stats['train_time']))
            print("Accuracy: {:.4}".format(layer.layer_stats['acc']))
            print("Loss: {:.4}".format(layer.layer_stats['loss']))
            print("")




class PassForwardThinking(object):
    """ A feed forward data transormation approach. """
    
    def __init__(self, layer_dims):
        """
        Args:
          layer_dims : (list)
        """
        self.model_version = '1.0'
        self.layers = []
        self.layer_dims = layer_dims
        self.num_classes = layer_dims[-1]

        input_dim = layer_dims[0]
        for l in layer_dims[1:]:
            new_layer = Layer(input_dim, l, num_classes=self.num_classes)
            input_dim += l
            self.layers.append(new_layer)

            
    def fit(self, x_train, y_train, epochs=25, reg_amnt=0, learning_rate=.01,
            batch_size=32, nonlinear='relu', loss_func='categorical_crossentropy',
            weight_scale=.1, early_stopping=False, reg_type=1, verbose=True):
        """ Fit the forward thinking model.
        Args:
          x_train : (array)
          y_train : (array)
          epochs : (int)
          reg_amnt : (float)
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

        if y_train.ndim == 1:
            y_train =  to_categorical(y_train) 
        if verbose: print("Starting Training for PushForwardThinking")
        t0 = time.time()
        for i, layer in enumerate(self.layers):
            if verbose: print("[ Training Layer %s ]" % i)
            layer.fit(x_train, y_train, epochs=epochs, reg_amnt=reg_amnt, learning_rate=learning_rate, 
                    batch_size=batch_size, nonlinear=nonlinear, loss_func=loss_func,
                      weight_scale=weight_scale, early_stopping=early_stopping, reg_type=reg_type, verbose=verbose)
            transformed_data = layer.transform_data(x_train)
            x_train = np.hstack((x_train, transformed_data))

        t1 = time.time()
        self.training_time = t1 - t0

            
    def predict(self, x_test):
        """ Predict with the trained model.
        Args:
          x_test : (array)
        """
        for i, layer in enumerate(self.layers[:-1]):
            transformed_data = layer.transform_data(x_test)
            x_test = np.hstack((x_test, transformed_data))
        final_layer = self.layers[-1]
        return np.argmax(final_layer.predict(x_test), axis=1)
    
    
    def summary(self, dataset='unknown'):
        """ Returns a dictionary summary of the model and training. 
        Can be used with store_results to save to database. 
        """
        output = {}
        output['model_name'] = 'PassForwardThinking'
        output['model_version'] = self.model_version
        output['num_layers'] = len(self.layer_dims)
        output['layer_dimensions'] = self.layer_dims 
        output['dataset'] = dataset
        output['num_instances'] = self.num_instances 
        output['num_features'] = self.num_features
        output['training_time'] = self.training_time
        output['accuracy'] = self.layers[-1].layer_stats['acc']
        output['loss'] = self.layers[-1].layer_stats['loss']
        output['parameters'] = self.params

        layer_accs = []
        layer_times = []
        layer_losses = []
        for i, layer in enumerate(self.layers):
            layer_times.append(layer.layer_stats['train_time'])
            layer_accs.append(layer.layer_stats['acc'])
            layer_losses.append(layer.layer_stats['loss'])

        output['layer_accuracy'] = layer_accs
        output['layer_losses'] = layer_losses
        output['layer_times'] = layer_times

        return output
        

