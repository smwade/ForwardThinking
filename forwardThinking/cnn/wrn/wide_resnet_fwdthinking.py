import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def save_weights(model, filename, stage, block):
    stage = str(stage)
    block = str(block)

    conv1_weights = model.get_layer('conv'+stage+'-'+block+'_i').get_weights()
    conv2_weights = model.get_layer('conv'+stage+'-'+block+'_ii').get_weights()

    np.savez(filename, W_conv1=conv1_weights[0], b_conv1=conv1_weights[1],
                       W_conv2=conv2_weights[0], b_conv2=conv2_weights[1])


    fc1_weights = model.get_layer('fc1').get_weights()
    fc2_weights = model.get_layer('fc2').get_weights()

    np.savez('fc_'+filename, W_fc1=fc1_weights[0], b_fc1=fc1_weights[1],
                       W_fc2=fc2_weights[0], b_fc2=fc2_weights[1])


def load_weights(filename):
    weights = np.load(filename)
    return [[weights["W_conv1"], weights["b_conv1"]], [weights["W_conv2"], weights["b_conv2"]]]

def load_fc_weights(filename):
    weights = np.load(filename)
    return [[weights["W_fc1"], weights["b_fc1"]], [weights["W_fc2"], weights["b_fc2"]]]

def load_conv_weights(filename):
    weights = np.load(filename)
    return [weights["W_conv"], weights["b_conv"]]

def residual_block(input_tensor, num_filters, stage, block, trainable=True, weights=[]):
    stage = str(stage)
    block = str(block)

    # module structure proposed in http://arxiv.org/abs/1603.05027
    if trainable:
        x = keras.layers.BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3,3), padding='same',
                    name='conv'+stage+'-'+block+'_i')(x)
        x = Dropout(.3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3,3), padding='same',
                name='conv'+stage+'-'+block+'_ii',
                kernel_initializer='zeros', bias_initializer='zeros')(x)
        x = keras.layers.add([input_tensor, x])
        return x
    else:
        if not weights:
            raise ValueError("weights list cannot be empty")

        x = keras.layers.BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3,3), padding='same', trainable=False,
                weights=weights[0],
                name='conv'+stage+'-'+block+'_i')(x)
        x = Dropout(.3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3,3), padding='same', trainable=False,
                weights=weights[1],
                name='conv'+stage+'-'+block+'_ii')(x)

        x = keras.layers.add([input_tensor, x])
        return x

def layer11(epochs):
    batch_size = 100
    num_classes = 10
    data_augmentation = True

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    main_input = Input(shape=x_train.shape[1:], name='main_input')

    conv1 = Conv2D(64, (3,3), activation='relu', name='conv1')(main_input)
    block1 = residual_block(conv1, 64, 1, 1)

    flat = Flatten()(block1)

    fc1 = Dense(512, activation='relu', name='fc1')(flat)
    fc1_drop = Dropout(.5)(fc1)
    main_output = Dense(10, activation='relu', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
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
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))

    print (np.argmax(model.predict(x_test), axis=1) == np.argmax(y_test, axis=1)).mean()

    conv1_weights = model.get_layer('conv1').get_weights()
    np.savez('conv1_weights.npz', W_conv=conv1_weights[0], b_conv=conv1_weights[1])

    return model

def layer12(epochs):
    batch_size = 100
    num_classes = 10
    data_augmentation = True

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    main_input = Input(shape=x_train.shape[1:], name='main_input')

    conv1 = Conv2D(64, (3,3), activation='relu', name='conv1', trainable=False)(main_input)
    block11 = residual_block(conv1, 64, 1, 1, trainable=False, weights=load_weights('weights_layer1.npz'))
    block12 = residual_block(block11, 64, 1, 2)

    flat = Flatten()(block12)

    fc_weights = load_fc_weights("fc_weights_layer1.npz")

    fc1 = Dense(512, activation='relu', weights=fc_weights[0], name='fc1')(flat)
    fc1_drop = Dropout(.5)(fc1)
    main_output = Dense(10, activation='relu', weights=fc_weights[1], name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.get_layer('conv1').set_weights(load_conv_weights('conv1_weights.npz'))


    """
    print "\n\n\nConv1"
    print model.get_layer('conv1').get_weights()
    print "\nConv1-1_i\n"
    print model.get_layer('conv1-1_i').get_weights()
    print "\nConv1-1_ii\n"
    print model.get_layer('conv1-1_ii').get_weights()
    print "\nfc1\n"
    print model.get_layer('fc1').get_weights()
    print "\nfc2\n"
    print model.get_layer('fc2').get_weights()
    """


    print (np.argmax(model.predict(x_test), axis=1) == np.argmax(y_test, axis=1)).mean()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
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
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))

    return model

def layer13(epochs):
    batch_size = 100
    num_classes = 10
    data_augmentation = True

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    main_input = Input(shape=x_train.shape[1:], name='main_input')

    conv1 = Conv2D(64, (3,3), activation='relu', name='conv1', trainable=False)(main_input)
    block11 = residual_block(conv1, 64, 1, 1, trainable=False, weights=load_weights('weights_layer1.npz'))
    block12 = residual_block(block11, 64, 1, 2, trainable=False, weights=load_weights('weights_layer2.npz'))
    block13 = residual_block(block12, 64, 1, 3)

    flat = Flatten()(block13)

    fc_weights = load_fc_weights("fc_weights_layer2.npz")

    fc1 = Dense(512, activation='relu', weights=fc_weights[0], name='fc1')(flat)
    fc1_drop = Dropout(.5)(fc1)
    main_output = Dense(10, activation='relu', weights=fc_weights[1], name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.get_layer('conv1').set_weights(load_conv_weights('conv1_weights.npz'))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
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
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))

    return model

def layer14(epochs):
    batch_size = 100
    num_classes = 10
    data_augmentation = True

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    main_input = Input(shape=x_train.shape[1:], name='main_input')

    conv1 = Conv2D(64, (3,3), activation='relu', name='conv1', trainable=False)(main_input)
    block11 = residual_block(conv1, 64, 1, 1, trainable=False, weights=load_weights('weights_layer1.npz'))
    block12 = residual_block(block11, 64, 1, 2, trainable=False, weights=load_weights('weights_layer2.npz'))
    block13 = residual_block(block12, 64, 1, 3, trainable=False, weights=load_weights('weights_layer3.npz'))
    block14 = residual_block(block13, 64, 1, 4)
    flat = Flatten()(block14)

    fc_weights = load_fc_weights("fc_weights_layer3.npz")

    fc1 = Dense(512, activation='relu', weights=fc_weights[0], name='fc1')(flat)
    fc1_drop = Dropout(0.5)(fc1)
    main_output = Dense(10, activation='relu', weights=fc_weights[1], name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.get_layer('conv1').set_weights(load_conv_weights('conv1_weights.npz'))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
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
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))

    return model


def layer15(epochs):
    batch_size = 100
    num_classes = 10
    data_augmentation = True

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    main_input = Input(shape=x_train.shape[1:], name='main_input')

    conv1 = Conv2D(64, (3,3), activation='relu', name='conv1', trainable=False)(main_input)
    block11 = residual_block(conv1, 64, 1, 1, trainable=False, weights=load_weights('weights_layer1.npz'))
    block12 = residual_block(block11, 64, 1, 2, trainable=False, weights=load_weights('weights_layer2.npz'))
    block13 = residual_block(block12, 64, 1, 3, trainable=False, weights=load_weights('weights_layer3.npz'))
    block14 = residual_block(block13, 64, 1, 4, trainable=False, weights=load_weights('weights_layer4.npz'))
    block15 = residual_block(block14, 64, 1, 5)

    flat = Flatten()(block15)

    #fc_weights = load_fc_weights("fc_weights_layer3.npz")

    fc1 = Dense(512, activation='relu', name='fc1')(flat)
    fc1_drop = Dropout(0.5)(fc1)
    main_output = Dense(10, activation='relu', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.get_layer('conv1').set_weights(load_conv_weights('conv1_weights.npz'))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
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
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))

    return model

def layer21(epochs):
    batch_size = 100
    num_classes = 10
    data_augmentation = True

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    main_input = Input(shape=x_train.shape[1:], name='main_input')

    conv1 = Conv2D(64, (3,3), activation='relu', name='conv1', trainable=False)(main_input)
    block11 = residual_block(conv1, 64, 1, 1, trainable=False, weights=load_weights('weights_layer1.npz'))
    block12 = residual_block(block11, 64, 1, 2, trainable=False, weights=load_weights('weights_layer2.npz'))
    block13 = residual_block(block12, 64, 1, 3, trainable=False, weights=load_weights('weights_layer3.npz'))
    block14 = residual_block(block13, 64, 1, 4, trainable=False, weights=load_weights('weights_layer4.npz'))
    block15 = residual_block(block14, 64, 1, 5)

    pool1 = MaxPooling2D(pool_size=(2,2))(block15)
    conv2 = Conv2D(128, (3,3), activation='relu', name='conv2')(pool1)
    block21 = residual_block(conv2, 128, 2, 1)

    flat = Flatten()(block21)

    #fc_weights = load_fc_weights("fc_weights_layer3.npz")

    fc1 = Dense(512, activation='relu', name='fc1')(flat)
    fc1_drop = Dropout(0.5)(fc1)
    main_output = Dense(10, activation='relu', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.get_layer('conv1').set_weights(load_conv_weights('conv1_weights.npz'))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
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
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))

    return model



if __name__ == "__main__":
    #save_weights(layer11(1), 'weights_layer1.npz', 1, 1)
    #save_weights(layer12(1), 'weights_layer2.npz', 1, 2)
    #save_weights(layer13(1), 'weights_layer3.npz', 1, 3)
    #save_weights(layer14(1), 'weights_layer4.npz', 1, 4)
    #save_weights(layer15(1), 'weights_layer5.npz', 1, 5)
    save_weights(layer21(1), 'weights_layer6.npz', 2, 1)
