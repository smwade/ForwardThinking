import numpy as np
from forwardThinking.datasets.toy_datasets import moons, chris_data, mnist_4_9, two_guassians
from forwardThinking.models import PushForwardThinking, DNN
from forwardThinking.visualize import plot_data, plot_decision

from matplotlib import pyplot as plt


# Import data
#x_train, y_train, x_test, y_test = two_guassians()
x_train, y_train, x_test, y_test = moons()

try:
    for i in range(1,10):
        x_train, y_train = chris_data(i, 200)

        print(" --- FORWARD THINKING ---")
        model = PushForwardThinking([2,100,50,2])
        model.fit(x_train, y_train, verbose=False, epochs=50)
        plt.subplot(121)
        plot_decision(x_train, y_train, model, title='ForwardThinking')

        print(" --- DNN ---")
        model = DNN([2,100,50,2])
        model.fit(x_train, y_train, verbose=False, epochs=50)
        plt.subplot(122)
        plot_decision(x_train, y_train, model, title='DNN')
        plt.show()
except:
    print("Complete.")
