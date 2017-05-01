from __future__ import division, print_function
import numpy as np

import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential

def relu(x):    
    return np.maximum(x, 0)

class PassForwardThinking(object):
    """ Keras based implementation of Pass-Forward Stacking. """

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.transform_weights = []


    def _build_layer_model(self, input_dim, hidden_dim, output_dim, frozen_weights, 
            activation='relu', loss_func='categorical_crossentropy', optimizer='adam'):
        """
         Build the layer model for one stage of passForwardThinking.

         Args:
           input_dim : (int) the size of the input data
           hidden_dim : (int) the number of nodes in hidden layer
           output_dim : (int) number of nodes in output layer
           frozen_weights : (list) list of frozen weights
                                    i.e [(input_dim, output_dim), (output_dim)]

        Returns:
          model : (keras.model) layer model
        """
        input_data = Input(shape=(input_dim,), name='input_data')

        # Old knowledge
        knowledge = Dense(output_dim, activation='sigmoid', 
                    name='knowledge_dense', trainable=False)(input_data)

        # Learn knowledge
        h1 = Dense(hidden_dim, activation=activation, kernel_initializer='zeros', 
                bias_initializer='zeros', name='transform_dense')(input_data)
        learn = Dense(output_dim, activation='sigmoid', kernel_initializer='zeros', 
                    bias_initializer='zeros', name='learn_dense')(h1)
        output = keras.layers.add([knowledge, learn], name='join')

        model = Model(inputs=input_data, outputs=output)
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
       
        # Freeze the learned weights
        model.get_layer('knowledge_dense').set_weights(frozen_weights)

        return model


    def _flatten_layer(self, layer_model):
        """ Flattens layer model by stacking knowledge and learn weights.
        Args:
          layer_model : (keras.model) A trained layer model

        Returns:
          [weights, bias] : (lsit) the modified weights and bias
        """
        old_weights, old_bias = layer_model.get_layer('knowledge_dense').get_weights()
        new_weights, new_bias = layer_model.get_layer('learn_dense').get_weights()
        print(old_weights.shape, new_weights.shape)
        weights = np.vstack((old_weights, new_weights))
        bias = old_bias + new_bias
        return [weights, bias]


    def fit(self, x_train, y_train, activation='relu', loss_func='categorical_crossentropy', 
            optimizer='adam', epochs=10, verbose=True):
        """ Train the model. 
        Args:
          x_train
          y_train
          activation
          loss_func
          optimizer
          epochs
        """
        # Inital Training
        input_dim = self.layer_dims[0]
        output_dim = self.layer_dims[-1]

        init_model = Sequential()
        init_model.add(Dense(output_dim, activation='sigmoid', input_shape=(input_dim,), name='init_dense'))
        init_model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
        init_model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
        frozen_weights = init_model.get_layer('init_dense').get_weights()
        

        acc_hist = []
        for i, layer_dim in enumerate(self.layer_dims[1:]):
            # Build and train layer
            if verbose: print("Training Layer %d" % i)
            layer = self._build_layer_model(input_dim, layer_dim, output_dim, frozen_weights,
                    activation=activation, loss_func=loss_func, optimizer=optimizer)

            layer.summary()
            layer_hist = layer.fit(x_train,  y_train, epochs=epochs, verbose=verbose)
            acc_hist += layer_hist.history['acc']

            # Transform input data
            t_weights = layer.get_layer('transform_dense').get_weights()
            self.transform_weights.append(t_weights)
            W, b = t_weights
            new_data = relu(np.dot(x_train, W) + b)
            x_train = np.hstack((x_train, new_data))# TODO check if the dimensions line up

            # Freeze the learned layer
            frozen_weights = self._flatten_layer(layer)

            input_dim += layer_dim

        # store final learned weights
        self.weights = frozen_weights
        return acc_hist

    def predict(x_test):

        # Transform the data
        for layer_weights in self.transform_weights:
            W, b = layer_weights
            x_test = np.dot(W, x) + b
            x_test = relu(x_test)

        # Classify
        # TODO
