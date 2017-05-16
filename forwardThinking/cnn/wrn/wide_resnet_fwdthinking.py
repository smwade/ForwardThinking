import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def save_weights(model, stage, block):
    conv1_weights = model.get_layer(stage+'-'+block+'_i').get_weights()
    conv2_weights = model.get_layer(stage+'-'+block+'_ii').get_weights()
    return [conv1_weights, conv2_weights]

def residual_block(input_tensor, num_filters, stage, block, pool=False, trainable=True, weights=[]):
    # module structure proposed in http://arxiv.org/abs/1603.05027
    if trainable:
        x = keras.layers.BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3,3), padding='same',
                    name=stage + '-' + block + '_i')(x)
        x = Dropout(.3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3,3), padding='same',
                name=stage + '-' + block + '_ii',
                kernel_initializer='zeros', bias_initializer='zeros')(x)
        x = keras.layers.add([input_tensor, x])
        return x
    else:
        if not weights:
            raise ValueError("weights list cannot be empty")

        def W_conv1(shape, dtype):
            if weights[0][0].shape != shape:
                raise ValueError()
            return weights[0][0].astype(dtype)
        def W_conv2(shape, dtype):
            if weights[0][0].shape != shape:
                raise ValueError()
            return weights[0][0].astype(dtype)
        def b_conv1(shape, dtype):
            if weights[0][1].shape != shape:
                raise ValueError()
            return weights[0][1].astype(dtype)
        def b_conv2(shape, dtype):
            if weights[1][1].shape != shape:
                raise ValueError()
            return weights[1][1].astype(dtype)

        x = keras.layers.BatchNormalization(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3,3), padding='same', trainable=False,
                kernel_initializer=W_conv1, bias_initializer=b_conv1,
                name=stage + block + '_i')(x)
        x = Dropout(.3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3,3), padding='same', trainable=False,
                kernel_initializer=W_conv2, bias_initializer=b_conv2,
                name=stage + block + '_ii')(x)

        x = keras.layers.add([input_tensor, x])
        return x

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

    conv1 = Conv2D(32, (3,3), activation='relu')(main_input)
    block1 = residual_block(conv1, 32, '1', '1')

    flat = Flatten()(block1)

    fc1 = Dense(512, activation='relu', name='fc1')(flat)
    fc1_drop = Dropout(.5)(fc1)
    main_output = Dense(10, activation='relu', name='fc2')(fc1_drop)

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
    block11 = save_weights(model, '1', '1')
    return conv1_weights, block11

conv1_weights, block11 = layer1(1)
