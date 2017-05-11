import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

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
    conv0 = Conv2D(16, (3,3), padding='same', activation='relu', name='conv1')(main_input)

    # Conv1
    conv111 = Conv2D(64, (3,3), padding='same', name='conv111')(conv0)
    conv111_drop = Dropout(.3)(conv111)
    conv112 = Conv2D(64, (3,3), padding='same', name='conv112')(conv111_drop)
    conv11_output = keras.layers.add([conv0, conv112], name='conv11_output')
    conv11_bn = keras.layers.BatchNormalization()(conv11_output)
    conv11_final = Activation('relu')(conv11_bn)

    conv121 = Conv2D(64, (3,3), padding='same', name='conv121')(conv11_final)
    conv121_drop = Dropout(.3)(conv121)
    conv122 = Conv2D(64, (3,3), padding='same', name='conv122')(conv121_drop)
    conv12_output = keras.layers.add([conv11_final, conv122], name='conv12_output')
    conv12_bn = keras.layers.BatchNormalization()(conv12_output)
    conv12_final = Activation('relu')(conv12_bn)

    # Conv2
    conv211 = Conv2D(128, (3,3), padding='same', name='conv211')(conv12_final)
    conv211_drop = Dropout(.3)(conv211)
    conv212 = Conv2D(128, (3,3), padding='same', name='conv212')(conv211_drop)
    conv21_output = keras.layers.add([conv12_final, conv212], name='conv21_output')
    conv21_bn = keras.layers.BatchNormalization()(conv21_output)
    conv21_final = Activation('relu')(conv21_bn)

    conv221 = Conv2D(128, (3,3), padding='same', name='conv221')(conv21_final)
    conv221_drop = Dropout(.3)(conv211)
    conv222 = Conv2D(128, (3,3), padding='same', name='conv222')(conv221_drop)
    conv22_output = keras.layers.add([conv21_final, conv222], name='conv22_output')
    conv22_bn = keras.layers.BatchNormalization()(conv22_output)
    conv22_final = Activation('relu')(conv22_bn)

    # Conv3
    conv311 = Conv2D(128, (3,3), padding='same', name='conv311')(conv22_final)
    conv311_drop = Dropout(.3)(conv311)
    conv312 = Conv2D(128, (3,3), padding='same', name='conv312')(conv311_drop)
    conv31_output = keras.layers.add([conv22_final, conv312], name='conv31_output')
    conv31_bn = keras.layers.BatchNormalization()(conv31_output)
    conv31_final = Activation('relu')(conv31_bn)

    conv321 = Conv2D(256, (3,3), padding='same', name='conv321')(conv31_final)
    conv321_drop = Dropout(.3)(conv311)
    conv322 = Conv2D(256, (3,3), padding='same', name='conv322')(conv321_drop)
    conv32_output = keras.layers.add([conv31_final, conv322], name='conv32_output')
    conv32_bn = keras.layers.BatchNormalization()(conv32_output)
    conv32_final = Activation('relu')(conv32_bn)

    # Avgpool

    avg_pool = AveragePooling2D(pool_size=(8,8))(conv32_final)
    main_output = Dense(10, activation='softmax')

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


layer1(1)
