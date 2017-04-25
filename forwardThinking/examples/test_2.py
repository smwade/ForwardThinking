""" Compare forwardThinking to passForwardThinking on toy_datasets """

import numpy as np
from forwardThinking.datasets.toy_datasets import moons, chris_data, mnist_4_9, two_guassians
from forwardThinking.models import PassForwardThinking, DNN, ForwardThinking
from forwardThinking.visualize import plot_data, plot_decision

from matplotlib import pyplot as plt

for i in range(1,7):
    x_train, y_train = chris_data(i, 1000)

    print("\n --- FORWARD THINKING ---")
    model = PassForwardThinking([2,100,50,10,2])
    model.fit(x_train, y_train, verbose=True, epochs=50)
    plt.subplot(121)
    plot_decision(x_train, y_train, model, title='PassForwardThinking')

    print(" --- DNN ---")
    plt.subplot(122)
    model = ForwardThinking([2,100,50,10,2])
    model.fit(x_train, y_train, verbose=True, epochs=50)
    plot_decision(x_train, y_train, model, title='ForwardThinking')
    plt.show()
