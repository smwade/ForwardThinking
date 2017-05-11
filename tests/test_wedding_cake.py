from forwardThinking.datasets import load_mnist
from forwardThinking.models import WeddingCake

x_train, y_train, x_test, y_test = load_mnist()

model = WeddingCake(784, 100, 10, 3)
model.fit(x_train, y_train, x_test, y_test, epochs=3)
