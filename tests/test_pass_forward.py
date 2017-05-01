from forwardThinking.datasets import load_mnist
from forwardThinking.models import PassForwardThinking

x_train, y_train, x_test, y_test = load_mnist()

model = PassForwardThinking([784, 100, 50, 10])
model.fit(x_train, y_train, epochs=2)
