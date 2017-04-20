import numpy as np
from matplotlib import pyplot as plt

def plot_data(x, y):
    plt.scatter(x[:,0], x[:,1], c=y)
    plt.show()

def plot_decision(X, Y, model, title=""):
    h = .02
    # Plotting decision regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))


    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.8)
    plt.title(title)

    #plt.show()


