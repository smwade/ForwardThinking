#wrn_test2.py
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

def residual_block(input_tensor, num_filters, pool=False):
    if pool:
        x = Conv2D(num_filters, (3,3), strides=(2,2), padding='same')(input_tensor)
    else:
        x = Conv2D(num_filters, (3,3), padding='same')(input_tensor)
    x = Dropout(.3)(x)
    x = Conv2D(num_filters, (3,3), padding='same')(x)
    x = keras.layers.add([input_tensor, x])
    x = keras.layers.BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def wide_resnet(epochs):
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

    # Conv1
    x = Conv2D(64, (7,7), padding='same', activation='relu')(main_input)

    # Conv2
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Conv3
    x = Conv2D(128, (3,3), strides=(2,2), activation='relu', padding='same')(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Conv4
    x = Conv2D(256, (3,3), strides=(2,2), activation='relu', padding='same')(x)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Avgpool
    avg_pool = AveragePooling2D(pool_size=(8,8))(x)
    avg_pool_flat = Flatten()(avg_pool)
    main_output = Dense(10, activation='softmax')(avg_pool_flat)

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
        import time
        before = time.time()
	model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))
        after = time.time()

        with open("wrn_results.txt", 'w') as f:
	    f.write(str(after - before) + '\n')
            f.write(str((np.argmax(model.predict(x_test), axis=1)==np.argmax(y_test, axis=1)).mean()) + '\n')
        
   
wide_resnet(200)