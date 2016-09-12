#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Cube26 product code
#
# (C) Copyright 2015 Cube26 Software Pvt Ltd
# All right reserved.
#
# This file is confidential and NOT open source.  Do not distribute.
#

"""

"""

import numpy as np
from scipy.io import loadmat
from utils import get_weights
from theano import function, shared
import theano.tensor as T
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def get_digits(n_samples=10000, size=8):
    images = loadmat('IMAGES.mat')["IMAGES"]
    imageix = np.random.randint(10, size=(n_samples,))
    patchix = np.random.randint(512 - size, size=(n_samples, 2))
    X = np.zeros((n_samples, size ** 2))
    for i in xrange(n_samples):
        xx, yy = patchix[i, :]
        X[i, :] = images[xx:(xx + size), yy:(yy + size), imageix[i]].ravel()
    return MinMaxScaler().fit_transform(X)


class Autoencoder(object):

    def __init__(self, layers, alpha=0.3, C=0.0, rho=0.05, beta=0.1):
        if len(layers) != 3:
            raise ValueError("Not an autoencoder")
        _x = T.dmatrix('x')
        _y = T.dmatrix('y')
        _lambda = C
        alpha = alpha
        # parameters
        w1 = shared(get_weights((layers[1], layers[0])), name="w1")
        w2 = shared(get_weights((layers[2], layers[1])), name="w2")
        b1 = shared(get_weights((layers[1], 1)), name="b1")
        b2 = shared(get_weights((layers[2], 1)), name="b2")

        a1 = _x.T
        z2 = T.dot(w1, a1) + b1.repeat(a1.shape[1], axis=1)
        a2 = 1.0 / (1 + T.exp(-z2))
        z3 = T.dot(w2, a2) + b2.repeat(a2.shape[1], axis=1)
        a3 = 1.0 / (1 + T.exp(-z3))
        self._predict = function([_x], [a3])

        loss = T.sum((a3 - _y.T) ** 2) / 0.5
        loss += _lambda / 2.0 * T.sum([(w1 ** 2).sum(), (w2 ** 2).sum()])
        # Add the KL divergence term
        rhohat = a2.sum(axis=1) / a2.shape[1]
        kl_divergence = (rho * T.log10(rho / rhohat)) + \
            (1 - rho) * T.log10((1 - rho) / (1 - rhohat))
        loss += beta * kl_divergence.sum()
        grad_w1, grad_b1 = T.grad(loss, [w1, b1])
        grad_w2, grad_b2 = T.grad(loss, [w2, b2])
        updates = [
                (w1, w1 - alpha * grad_w1 / _x.shape[0]),
                (w2, w2 - alpha * grad_w2 / _x.shape[0]),
                (b1, b1 - alpha * grad_b1 / _x.shape[0]),
                (b2, b2 - alpha * grad_b2 / _x.shape[0])]
        self._train = function([_x, _y], [loss], updates=updates)
        self.w1 = w1

    def predict(self, X):
        return self._predict(X)[0].T

    def fit(self, X, n_iter=1000, showloss=False):
        self.predict(X)
        losses = []
        for i in xrange(n_iter):
            losses.append(self._train(X, X)[0])
            if showloss:
                print i, losses[-1]

    def visualize(self, nrows, ncols, show=True, **kwargs):
        w = self.w1.get_value()
        divisors = np.sqrt((w ** 2).sum(1))
        w = w.T / divisors
        w = w[:(nrows * ncols)]
        fig, ax = plt.subplots(nrows, ncols)
        for i in xrange(nrows):
            for j in xrange(ncols):
                ax[i, j].imshow(w[:, i + 5 * j].reshape(5, 5),
                                cmap=plt.cm.gray, **kwargs)
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
                ax[i, j].grid(False)
        if show:
            plt.show()


if __name__ == '__main__':
    X = get_digits()
    aenc = Autoencoder(alpha=0.3, layers=[64, 25, 64], rho=0.01,
            C=0.0001, beta=3.0)
    aenc.fit(X, n_iter=1000, showloss=True)
    aenc.visualize(5, 5, interpolation="nearest")
