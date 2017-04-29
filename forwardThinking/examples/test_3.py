import numpy as np
from forwardThinking.datasets.toy_datasets import load_cifar10
from forwardThinking.models import PassForwardThinking, DNN, store_results
from forwardThinking.visualize import plot_data, plot_decision

from matplotlib import pyplot as plt

# Import data
x_train, y_train, x_test, y_test = load_cifar10()

model = PassForwardThinking([3072,500,100,10], stack_data=True)
model.fit(x_train, y_train, verbose=True, epochs=10)
plt.plot(model.training_history(), label='passForwardThinking')
store_results(model, "cifar_10")


model = PassForwardThinking([3072,500,100,10], stack_data=False)
model.fit(x_train, y_train, verbose=True, epochs=10)
plt.plot(model.training_history(), label='forwardThinking')
store_results(model, "cifar_10")

model = DNN([3072,500,100,10])
model.fit(x_train, y_train, verbose=True, epochs=30)
plt.plot(model.training_history(), label='dnn')
plt.legend()
plt.title("Acc Over Time")
plt.show()
store_results(model, "cifar_10")
