import os
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import keras.backend as K

import time

from scipy.stats import truncnorm

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

VERBOSE = False
batch_size = 32
num_classes = 10
data_augmentation = True

number_of_blitz = 2

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
if VERBOSE:
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

best_score = 0
train_begin_time = time.time()

def _identity_with_noise(shape, dtype=np.float32):
    tnorm = truncnorm(a=-.01, b=.01, loc=0., scale=0.0001)
    output = tnorm.rvs(np.product(shape)).reshape(shape).astype(dtype)
    for i in xrange(output.shape[2]):
        output[1,1,i,i] = 1.
    return output

def save_weights(model, filename, layer):
    conv1a = model.get_layer('conv{0}a'.format(layer)).get_weights()
    conv1b = model.get_layer('conv{0}b'.format(layer)).get_weights()
    fc1 = model.get_layer('fc1').get_weights()
    fc2 = model.get_layer('fc2').get_weights()

    np.savez(filename, W_conva=conv1a[0], b_conva=conv1a[1],
                W_convb=conv1b[0], b_convb=conv1b[1],
                W_fc1=fc1[0], b_fc1=fc1[1],
                W_fc2=fc2[0], b_fc2=fc2[1])

def load_fc_weights(filename):
    weights = np.load(filename)
    return [[weights["W_fc1"], weights["b_fc1"]], [weights["W_fc2"], weights["b_fc2"]]]

def load_conv_weights(filename):
    weights = np.load(filename)
    return [[weights["W_conva"], weights["b_conva"]],
            [weights["W_convb"], weights["b_convb"]]]

def load_prelu_weights(filename):
    weights = np.load(filename)
    return weights['prelu']

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.epoch_times = []
        self.best_weights = None

    def on_epoch_begin(self, epoch, logs={}):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs={}):
        global best_score
        self.times.append(time.time() - train_begin_time)
        self.epoch_times.append(time.time() - self.t0)

        if logs.get('val_acc') > best_score:
            try:
                best_score = logs.get('val_acc')
                self.best_weights = save_weights(self.model, 'weights_layer3.npz', 3)
            except Exception:
                pass

def blitz(epochs):
    main_input = Input(shape=x_train.shape[1:], name='main_input')
    x = Conv2D(512, (3,3),
                activation='relu',
                   padding='same',
                   name='conv1a')(main_input)
    x = Conv2D(128, (3,3),
                    activation='relu',
                   padding='same',
                   name='conv1b')(x)
    x = MaxPooling2D(pool_size=(4,4))(x)
    x = Dropout(0.3)(x) ####

    flat = Flatten()(x)

    fc1 = Dense(512, activation='relu', name='fc1')(flat)
    fc1_drop = Dropout(0.5)(fc1)
    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer=keras.optimizers.Adam(lr=0.002),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if not data_augmentation:
        if VERBOSE: print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        if VERBOSE: print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        time_history = TimeHistory()
        history = keras.callbacks.History()
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                                steps_per_epoch=x_train.shape[0] // batch_size,
                                                epochs=epochs, callbacks=[history, time_history],
                                                validation_data=(x_test, y_test))

        np.savez('blitz_results.npz', acc=history.history['acc'], loss=history.history['loss'],
                          val_acc=history.history['val_acc'], val_loss=history.history['val_loss'],
                  times=time_history.times, epoch_times=time_history.epoch_times)

    return model

def layer1(epochs):
    conv_weights1 = load_conv_weights('weights1.npz')

    main_input = Input(shape=x_train.shape[1:], name='main_input')
    x = Conv2D(512, (3,3),
                activation='relu',
                   padding='same',
                   name='conv1a',
                   weights=conv_weights1[0],
                   trainable=False)(main_input)
    x = Conv2D(128, (3,3),
                    activation='relu',
                   padding='same',
                   name='conv1b',
                   weights=conv_weights1[1],
                   trainable=False)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    #x = Dropout(0.3)(x) ####

    x = Conv2D(512, (3,3),
                activation='relu',
                   padding='same',
                   name='conv2a')(x)
    x = Conv2D(128, (3,3),
                    activation='relu',
                   padding='same',
                   name='conv2b')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x) ####

    flat = Flatten()(x)

    fc1 = Dense(512, activation='relu', name='fc1')(flat)
    fc1_drop = Dropout(0.5)(fc1)
    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer=keras.optimizers.Adam(lr=0.003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if not data_augmentation:
        if VERBOSE: print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        if VERBOSE: print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        time_history = TimeHistory()
        history = keras.callbacks.History()
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                                steps_per_epoch=x_train.shape[0] // batch_size,
                                                epochs=epochs, callbacks=[history, time_history],
                                                validation_data=(x_test, y_test))

        np.savez('blitz_results.npz', acc=history.history['acc'], loss=history.history['loss'],
                          val_acc=history.history['val_acc'], val_loss=history.history['val_loss'],
                  times=time_history.times, epoch_times=time_history.epoch_times)

    return model

def layer2(epochs):
    conv_weights1 = load_conv_weights('weights1.npz')
    conv_weights2 = load_conv_weights('weights2.npz')

    main_input = Input(shape=x_train.shape[1:], name='main_input')
    x = Conv2D(512, (3,3),
                activation='relu',
                   padding='same',
                   name='conv1a',
                   weights=conv_weights1[0],
                   trainable=False)(main_input)
    x = Conv2D(128, (3,3),
                    activation='relu',
                   padding='same',
                   name='conv1b',
                   weights=conv_weights1[1],
                   trainable=False)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(512, (3,3),
                activation='relu',
                   padding='same',
                   name='conv2a',
                   weights=conv_weights2[0],
                   trainable=False)(x)
    x = Conv2D(128, (3,3),
                    activation='relu',
                   padding='same',
                   name='conv2b',
                   weights=conv_weights2[1],
                   trainable=False)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x) ####

    x = Conv2D(512, (3,3),
                activation='relu',
                   padding='same',
                   name='conv3a')(x)
    x = Conv2D(128, (3,3),
                    activation='relu',
                   padding='same',
                   name='conv3b')(x)
    x = Dropout(0.3)(x)

    flat = Flatten()(x)

    fc1 = Dense(512, activation='relu', name='fc1')(flat)
    fc1_drop = Dropout(0.5)(fc1)
    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    def schedule(epoch):
        if epoch < 20:
 	        return 0.001
        elif epoch < 70:
            return 0.0005
        else:
            return 0.0001

    rate_schedule = keras.callbacks.LearningRateScheduler(schedule)

    if not data_augmentation:
        if VERBOSE: print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        if VERBOSE: print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        time_history = TimeHistory()
        history = keras.callbacks.History()
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                                steps_per_epoch=x_train.shape[0] // batch_size,
                                                epochs=epochs, callbacks=[history, time_history, rate_schedule],
                                                validation_data=(x_test, y_test))

        np.savez('blitz_results.npz', acc=history.history['acc'], loss=history.history['loss'],
                          val_acc=history.history['val_acc'], val_loss=history.history['val_loss'],
                  times=time_history.times, epoch_times=time_history.epoch_times)

    return model


if __name__ == "__main__":
    save_weights(blitz(20), 'weights1.npz', 1)
    K.clear_session()
    save_weights(layer1(20), 'weights2.npz', 2)
    K.clear_session()

    save_weights(layer2(100), 'weights3.npz', 3)
