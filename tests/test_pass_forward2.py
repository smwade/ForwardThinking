from forwardThinking.datasets import load_mnist
from forwardThinking.models import PassForwardThinking, AdaptiveForwardThinking
import numpy as np

x_train, y_train, x_test, y_test = load_mnist()
x_train = np.load('aug_mnist_data2.npy')
y_train = np.load('aug_mnist_labels2.npy')

model = PassForwardThinking([784, 150, 100, 100, 50, 10], freeze=True)
model.fit(x_train, y_train, x_test, y_test, epochs=10, reg_type='l2', reg=.00001)
