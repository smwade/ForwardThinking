import numpy as np

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils.np_utils import to_categorical


class PassForwardThinking(object):
    """ Keras based implementation of Pass-Forward Stacking. """

    def __init__(self):
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
        h1 = Dense(hidden_dim, activation=activation, kernal_initializer='zeros', 
                bias_initializer='zeros', name='transform_dense')(input_data)
        learn = Dense(output_dim, activation='sigmoid', kernal_initializer='zeros', 
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
        weights = np.vstack((old_weights, new_weights))
        bias = np.hstack((old_bias, new_bias))
        return [weights, bias]

    def fit(self, x_train, y_train, activation='relu', loss_func='categorical_crossentropy', 
            optimizer='adam', epochs=10):
        NUM_LAYERS = 3
        HIDDEN_DIM = 4

        # Inital Training
        init_model = Sequential()
        init_model.add(Dense(output_dim, activation='sigmoid', input_shape=(input_dim,), name='init_dense'))
        init_model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
        init_model.fit(x_train, y_train, epochs=epochs)
        frozen_weights = init_model.get_layer('init_dense').get_weights()

        while layer_num < NUM_LAYERS:

            # Build and train layer
            layer = self._build_layer_model(input_dim, HIDDEN_DIM, output_dim, frozen_weights)
            layer.fit(x_train,  y_train)

            # Transform input data
            t_weights = layer.get_layer('transform_dense').get_weights()
            self.transform_weights.append(t_weights)
            W, b = t_weights
            x_train = np.dot(W, x_train) + b # TODO check if the dimensions line up

            # Freeze the learned layer
            frozen_weights = self._flatten_layer(layer)

            layer_num += 1

        # store final learned weights
        self.weights = frozen_weights

    def predict(x_test):

        # Transform the data
        for layer in self.transform_weights:
            W, b = layer
            x_test = np.dot(W, x) + b
            x_test = relu(x_test)

        # Classify





if __name__ == '__main__':
    a = PassForwardThinking()
    a.fit(None, None)

