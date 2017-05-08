from forwardThinking.datasets import load_mnist
from forwardThinking.models import AdaptiveForwardThinking

x_train, y_train, x_test, y_test = load_mnist()

#model = PassForwardThinking([784, 100, 100, 10], freeze=False)
#model.fit(x_train, y_train, x_test, y_test, epochs=100)

model = AdaptiveForwardThinking(784, 1, 10, freeze=False)
model.fit(x_train, y_train, x_test, y_test, max_epochs=100)
