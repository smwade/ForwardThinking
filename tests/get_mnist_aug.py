from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from itertools import product


def single_pixel_tr(X, y, img_size=(28,28)):
    n_samples, dim_data = X.shape
    if img_size[0]*img_size[1] != dim_data: # must make sure dim of data is square 
        raise ValueError('img_size not compatible with size of data in X')
    X_out = np.empty((n_samples*9, dim_data))
    y_out = np.zeros(n_samples*9)
    for k in xrange(n_samples):
        k_image = X[k,:].reshape(img_size)
        for i,j in product([-1,0,1], [-1,0,1]):
            i_row = 3*(i+1)
            image = np.zeros((img_size[0]+2, img_size[1]+2))
            image[1+i:1+i+img_size[0], 1+j:1+j+img_size[1]] = k_image
            X_out[9*k+i_row+j+1] = image[1:1+img_size[0],1:1+img_size[1] ].flatten()
            y_out[9*k+i_row+j+1] = y[k]
    return X_out, y_out


def get_MNIST_aug():
	# get the MNIST data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	D_train = mnist.train
	D_test = mnist.test
	X_train = D_train.images
	y_train = np.argmax(D_train.labels, axis=1)
	X_t = D_test.images
	y_t = D_test.labels
	y_t = np.argmax(y_t, axis=1)
	X_train, y_train = single_pixel_tr(X_train,y_train)
	MNIST = {}
	MNIST['X_train'], MNIST['y_train'] = X_train, y_train
	MNIST['X_test'], MNIST['y_test'] = X_t, y_t
	return MNIST



