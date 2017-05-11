import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def layer1(epochs):
    batch_size = 32
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
    conv1 = Conv2D(32, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv1')(main_input)
    conv1_flat = Flatten()(conv1)
    fc1 = Dense(512, activation='relu', name='fc1')(conv1_flat)
    fc1_drop = Dropout(0.5)(fc1)
    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam',
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

    conv1_weights = model.get_layer('conv1').get_weights()
    return conv1_weights


def layer2(epochs, conv1_weights):
    batch_size = 32
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
    conv1 = Conv2D(32, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv1',
                   trainable=False)(main_input)
    data_concat = keras.layers.concatenate([main_input, conv1], axis=3)

    conv2 = Conv2D(32, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv2')(data_concat)
    conv2_drop = Dropout(.35)(conv2)
    conv2_flat = Flatten()(conv2_drop)

    fc1 = Dense(512, activation='relu', name='fc1')(conv2_flat)
    fc1_drop = Dropout(.5)(fc1)
    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.get_layer('conv1').set_weights(conv1_weights)

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

    conv1_weights = model.get_layer('conv1').get_weights()
    conv2_weights = model.get_layer('conv2').get_weights()

    return conv1_weights, conv2_weights

def layer3(epochs, conv1_weights, conv2_weights):
    batch_size = 32
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
    conv1 = Conv2D(32, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv1',
                   trainable=False)(main_input)
    data_concat = keras.layers.concatenate([main_input, conv1], axis=3)
    conv2 = Conv2D(32, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv2',
                   trainable=False)(data_concat)

    main_pool = MaxPooling2D(pool_size=(2,2))(main_input)
    conv1_pool = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2_pool = MaxPooling2D(pool_size=(2,2))(conv2)

    main_drop = Dropout(.25)(main_pool)
    conv1_drop = Dropout(.25)(conv1_pool)
    conv2_drop = Dropout(.25)(conv2_pool)

    final_input = keras.layers.concatenate([main_drop, conv1_drop, conv2_drop], axis=3)

    conv3 = Conv2D(64, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv3')(final_input)
    conv3_flat = Flatten()(conv3)

    fc1 = Dense(512, activation='relu', name='fc1')(conv3_flat)
    fc1_drop = Dropout(.5)(fc1)
    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.get_layer('conv1').set_weights(conv1_weights)
    model.get_layer('conv2').set_weights(conv2_weights)

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

    conv1_weights = model.get_layer('conv1').get_weights()
    conv2_weights = model.get_layer('conv2').get_weights()
    conv3_weights = model.get_layer('conv3').get_weights()
    fc1_weights = model.get_layer('fc1').get_weights()
    fc2_weights = model.get_layer('fc2').get_weights()

    return conv1_weights, conv2_weights, conv3_weights, fc1_weights, fc2_weights

def layer4(epochs, conv1_weights, conv2_weights, conv3_weights, fc1_weights, fc2_weights):
    batch_size = 32
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
    conv1 = Conv2D(32, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv1',
                   trainable=False)(main_input)
    data_concat = keras.layers.concatenate([main_input, conv1], axis=3)
    conv2 = Conv2D(32, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv2',
                   trainable=False)(data_concat)

    main_pool = MaxPooling2D(pool_size=(2,2))(main_input)
    conv1_pool = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2_pool = MaxPooling2D(pool_size=(2,2))(conv2)

    data_concat2 = keras.layers.concatenate([main_pool, conv1_pool, conv2_pool], axis=3)

    conv3 = Conv2D(64, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv3',
                   trainable=False)(data_concat2)

    data_concat3 = keras.layers.concatenate([main_pool, conv1_pool, conv2_pool, conv3], axis=3)

    conv4 = Conv2D(64, (3,3),
                   activation='relu',
                   padding='same',
                   name='conv4')(data_concat3)

    conv4_pool = MaxPooling2D(pool_size=(2,2))(conv4)
    conv4_drop = Dropout(.25)(conv4_pool)

    conv4_flat = Flatten()(conv4_drop)

    conv4_fc1 = Dense(512, activation='relu',
                        name='conv4_fc1')(conv4_flat)

    conv4_fc1_drop = Dropout(.5)(conv4_fc1)
    conv4_fc2 = Dense(10, activation='linear', name='conv4_fc2')(conv4_fc1_drop)

    """
    conv3_flat = Flatten()(conv3)
    conv3_fc1 = Dense(512, activation='linear',
                        name='conv3_fc1')(conv3_flat)
    conv3_fc1_drop = Dropout(.5)(conv3_fc1)
    conv3_fc2 = Dense(10, activation='linear', name='conv3_fc2')(conv3_fc1_drop)

    fc2 = keras.layers.add([conv4_fc2, conv3_fc2])
    """
    main_output = Activation('softmax')(conv4_fc2)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.get_layer('conv1').set_weights(conv1_weights)
    model.get_layer('conv2').set_weights(conv2_weights)
    model.get_layer('conv3').set_weights(conv3_weights)
    model.get_layer('conv3_fc1').set_weights(fc1_weights)
    model.get_layer('conv3_fc2').set_weights(fc2_weights)

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



conv1_weights = layer1(1)
conv1_weights, conv2_weights = layer2(2, conv1_weights)
conv1_weights, conv2_weights, conv3_weights, fc1_weights, fc2_weights = layer3(3, conv1_weights, conv2_weights)

layer4(40, conv1_weights, conv2_weights, conv3_weights, fc1_weights, fc2_weights)
