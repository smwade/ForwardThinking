from forwardThinking.datasets import load_mnist
from forwardThinking.models import PassForwardThinking

x_train, y_train, x_test, y_test = load_mnist()

model = PassForwardThinking([784, 100, 100, 10, 100, 1, 1, 1, 1,10], freeze=False)
model.fit(x_train, y_train, x_test, y_test, epochs=2)
