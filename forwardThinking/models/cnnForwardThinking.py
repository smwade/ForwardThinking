from __future__ import division, print_function
import numpy as np
from time import time

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


class CnnForwardThinking(object):
    """ Convolution neural network forward thinking """

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.transform_weights = []
        self.summary = {}
        self.summary['model_name'] = 'CNNForwardThinking'
        self.summary['model_version'] = '1.0'


    def _build_layer_model(self, input_dim, conv_dim, output_dim,):
        """
        input_dim : (w,h,d) (32,32,3)
        conv_dim : (w,h,d)
        output_dim : intern
        """
        data_input = Input(shape=(32,32,3), name='data_input')
        kw, kh, kd = conv_dim
        conv = Conv2D(kd, (kw, kw), activation='relu', padding='same', name='conv')(data_input)
        conv_flat = Flatten()(conv)
        fc1 = Dense(512, activation='relu', name='fc1')(conv_flat)
        out = Dense(10, activation='softmax', name='fc2')(fc1)
    
        model = Model(inputs=[main_input], outputs=[main_output])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
        bias = old_bias + new_bias
        return [weights, bias]


    def fit(self, x_train, y_train, x_test, y_test, activation='relu', loss_func='categorical_crossentropy', 
            optimizer='adam', epochs=10, reg_type='l1', reg=0, dropout=False, verbose=True):
        """ Train the model. 
        Args:
          x_train
          y_train
          activation
          loss_func
          optimizer
          epochs
        """
        if verbose: print("Starting Training...\n")
        
        # Regularization
        if reg_type == 'l1':
            r = regularizers.l1(reg)
        if reg_type == 'l2':
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
        raise NotImplementedError("Just use training...")
