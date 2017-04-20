from __future__ import print_function, division

import numpy as np
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


    def fit(self, x_train, y_train, epochs=25, reg=0, learning_rate=.01,
            batch_size=32, nonlinear='relu', loss_func='categorical_crossentropy', 
            weight_scale=.1, early_stopping=False, reg_type=1, verbose=True):
        """ Train the layer. """

        if reg_type == 1:
            layer_reg = regularizers.l1(reg)
        else:
            layer_reg = regularizers.l2(reg)

        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        num_classes = self.num_classes

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
        return np.maximum(h, 0) # relu
    

class PushForwardThinking(object):
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
            input_dim += l
            self.layers.append(new_layer)

            
    def fit(self, x_train, y_train, epochs=25, reg=0, learning_rate=.01,
            batch_size=32, nonlinear='relu', loss_func='categorical_crossentropy',
            weight_scale=.1, early_stopping=False, reg_type=1, verbose=True):
        """ Fit the forward thinking model.
        Args:
          x_train : (array)
          y_train : (array)
          epochs : (int)
          reg : (float)
          learning_rate : (float)
          batch_size : (int)
          nonlinear : (string)
          weight_scale : (float)
        """  
        if y_train.ndim == 1:
            y_train =  to_categorical(y_train) 
        print(y_train.shape)
        print("Starting Training for PushForwardThinking")
        for i, layer in enumerate(self.layers):
            print("[ Training Layer %s ]" % i)
            layer.fit(x_train, y_train, epochs=epochs, reg=reg, learning_rate=learning_rate, 
                    batch_size=batch_size, nonlinear=nonlinear, loss_func=loss_func,
                      weight_scale=weight_scale, early_stopping=early_stopping, reg_type=reg_type, verbose=verbose)
            transformed_data = layer.transform_data(x_train)
            x_train = np.hstack((x_train, transformed_data))

            
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

if __name__ == '__main__':
    from keras.datasets import mnist
    import keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    ft = PushForwardThinking([784, 5, 5, 10])
    ft.fit(x_train, y_train)

