import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Activation, Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Model
from keras import backend as K


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def layer1():
    main_input = Input(shape=input_shape, name='main_input')
    conv1 = Conv2D(32,(3,3), activation='relu', padding='same', name='conv1')(main_input)
    conv1_drop = Dropout(.25)(conv1)
    conv1_flat = Flatten()(conv1_drop)
    fc1 = Dense(128, activation='relu', name='fc1')(conv1_flat)
    fc1_drop = Dropout(.5)(fc1)
    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=100, validation_data=(x_test, y_test))

    conv1_weights = model.get_layer('conv1').get_weights()
    fc1_weights = model.get_layer('fc1').get_weights()
    fc2_weights = model.get_layer('fc2').get_weights()
    return conv1_weights, fc1_weights, fc2_weights

conv1_weights, fc1_weights, fc2_weights = layer1()

def layer2(conv1_weights, fc1_weights, fc2_weights):
    main_input = Input(shape=input_shape, name='main_input')
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same', trainable=False, name='conv1')(main_input)
    data_concat = keras.layers.concatenate([main_input, conv1], axis=3)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2')(data_concat)
    
    conv1_drop = Dropout(.35)(conv1)
    conv1_flat = Flatten()(conv1_drop)

    conv2_max = MaxPooling2D(pool_size=(2,2))(conv2)
    conv2_drop = Dropout(.35)(conv2_max)
    conv2_flat = Flatten()(conv2_drop)

    fc_conv1 = Dense(128, activation='linear', name='fc_conv1')(conv1_flat)
    fc_conv2 = Dense(128, kernel_initializer='zeros', bias_initializer='zeros', activation='linear', name='fc_conv2')(conv2_flat)
    fc1 = keras.layers.add([fc_conv1, fc_conv2])
    fc1_relu = Activation('relu')(fc1)
    fc1_drop = Dropout(.5)(fc1_relu)

    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.get_layer('conv1').set_weights(conv1_weights)
    model.get_layer('fc_conv1').set_weights(fc1_weights)
    model.get_layer('fc2').set_weights(fc2_weights)
    model.fit(x_train, y_train, epochs=10, batch_size=100, validation_data=(x_test, y_test))
   
layer2(conv1_weights, fc1_weights, fc2_weights) 
