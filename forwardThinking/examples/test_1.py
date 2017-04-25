import numpy as np
from forwardThinking.datasets.toy_datasets import moons, chris_data, mnist_4_9, two_guassians
from forwardThinking.models import PassForwardThinking, DNN
from forwardThinking.visualize import plot_data, plot_decision

from matplotlib import pyplot as plt


# Import data
x_train, y_train, x_test, y_test = two_guassians()


print("\n --- FORWARD THINKING ---")
model = PassForwardThinking([2,100,10,2])
model.fit(x_train, y_train, verbose=True, epochs=50)
plt.subplot(121)
plot_decision(x_train, y_train, model, title='ForwardThinking')

print(" --- DNN ---")
plt.subplot(122)
model = DNN([2,100,10,2])
model.fit(x_train, y_train, verbose=True, epochs=50)
plot_decision(x_train, y_train, model, title='DNN')
plt.show()
