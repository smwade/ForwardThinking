""" A collection of useful datasets """
import os
import numpy as np
from sklearn.model_selection import train_test_split

def chris_data(indx=0, num_points=10000):
    dim = 2
    X = np.random.rand(num_points, dim)-1
    Y_list = [np.sign(X[:,0]+.3*X[:,1]-.2),
            np.sign(X[:,0])*np.sign(X[:,1]),
            np.sign(X[:,1]-(X[:,0]-.2)**2+.3),
            np.sign(X[:,0]**2+X[:,1]**2-.8**2),
            np.sign(X[:,0]-.7*np.sin(4*X[:,1])),
            np.sign(np.cos(5*X[:,0])-np.cos(3*X[:,0])*np.sin(4*X[:,1])),
            np.minimum(np.sign((X[:,0]+.5)**2+(X[:,1]+.5)**2-.4**2),np.sign((X[:,0]-.5)**2+(X[:,1]-.5)**2-.3**2))
            ]
    Y = Y_list[indx]
    Y[Y == -1] = 0
    return X, Y

def two_guassians(num_points=5000):
    np.random.seed(12)
    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_points)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_points)

    data = np.vstack((x1, x2)).astype(np.float32)
    labels = np.hstack((np.zeros(num_points),
                                np.ones(num_points)))

    x_train, x_test, y_train, y_test = train_test_split(data, labels)
    return x_train, y_train, x_test, y_test


def interior_circle(num_points=5000):
    from sklearn.datasets import make_hastie_10_2

    data, labels = make_hastie_10_2(n_samples=num_points, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(data, labels)
    return x_train, y_train, x_test, y_test


def moons(num_points=5000):
    from sklearn.datasets import make_moons

    data, labels = make_moons(n_samples=num_points, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(data, labels)
    return x_train, y_train, x_test, y_test


def mnist_4_9():
    this_dir, this_file = os.path.split(__file__)
    data_path = os.path.join(this_dir, "data",  "mnist_4_9.npz")
    data = np.load(data_path)
    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']
    return x_train, y_train, x_test, y_test

def load_iris():
    import sklearn.datasets as sk_data
    from sklearn.model_selection import train_test_split
    from keras.utils import np_utils
    data_dict = sk_data.load_iris()
    x_train, x_test, y_train_num, y_test_num = \
            train_test_split(data_dict['data'], data_dict['target'])

    y_train = np_utils.to_categorical(y_train_num, 3)
    y_test = np_utils.to_categorical(y_test_num, 3)
    return x_train, y_train, x_test, y_test
    


def load_mnist(flatten=True, one_hot_labels=True):
    """ Load mnist dataset using keras."""
    from keras.datasets import mnist
    from keras.utils import np_utils

    (x_train, y_train_num), (x_test, y_test_num) = mnist.load_data()

    if flatten:
        flatten_shape = np.prod(x_train.shape[1:])
        x_train = x_train.reshape(x_train.shape[0], flatten_shape).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], flatten_shape).astype('float32')
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train_num, 10)
    y_test = np_utils.to_categorical(y_test_num, 10)

    if not one_hot_labels:
        y_train = y_train_num
        y_test = y_test_num

    return x_train, y_train, x_test, y_test


def load_cifar10(flatten=True, one_hot_labels=True):
    """ Load cifar10 dataset with keras."""
    from keras.datasets import cifar10
    from keras.utils import np_utils

    (x_train, y_train_num), (x_test, y_test_num) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train_num, 10)
    y_test = np_utils.to_categorical(y_test_num, 10)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    y_train = np_utils.to_categorical(y_train_num, 10)
    y_test = np_utils.to_categorical(y_test_num, 10)

    if not one_hot_labels:
        y_train = y_train_num
        y_test = y_test_num

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    X, Y = chris()
    plt.scatter(X[:,0],X[:,1],c=.75*Y,vmin=-1,vmax=1)
    plt.show()


