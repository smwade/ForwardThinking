from forwardThinking.datasets import load_mnist

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

# Load data
x_train, y_train, x_test, y_test = load_mnist()

def my_init(shape, dtype=None, helper=True):
    if helper:
        print "HELO"
    return K.random_normal(shape, dtype=dtype)


model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(784,)))
model.add(Dense(64, kernel_initializer=my_init))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train)
