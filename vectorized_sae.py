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
Simple vectorized autoencoder
"""

import numpy as np
from utils import get_weights
from autoencoder import get_digits
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class Autoencoder(object):

    def __init__(self, layers, alpha=0.3, beta=3.0, C=0.0001, rho=0.01):
        self.wdecay = C
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.w1 = get_weights((layers[1], layers[0]))
        self.b1 = get_weights((layers[1], 1))
        self.w2 = get_weights((layers[2], layers[1]))
        self.b2 = get_weights((layers[2], 1))

    def predict(self, X):
        a1 = X.T
        z2 = np.dot(self.w1, a1) + self.b1
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.w2, self.a2) + self.b2
        return sigmoid(z3)

    def kl_divergence(self, rhohat):
        kld = self.rho * np.log(self.rho / rhohat) + \
            (1 - self.rho) * np.log((1 - self.rho) / (1 - rhohat))
        return kld.sum()

    def loss(self, X):
        loss = 0
        self.a3 = self.predict(X)
        loss += 0.5 * ((self.a3 - X.T) ** 2).sum() / X.shape[0]
        theta = np.r_[self.w1.ravel(), self.w2.ravel()]
        loss += self.wdecay / 2.0 * (theta ** 2).sum()
        self.rhohat = self.a2.sum(1) / X.shape[0]
        loss += self.beta * self.kl_divergence(self.rhohat)
        return loss

    def fit(self, X, n_iter=400, showloss=True):
        losses = []
        for i in xrange(n_iter):
            loss = self.loss(X)
            losses.append(loss)
            if showloss:
                print i, loss

            # Update gradients
            del3 = -(X.T - self.a3) * self.a3 * (1 - self.a3)
            del2 = np.dot(self.w2.T, del3)
            del2 += self.beta * ((-self.rho / self.rhohat) + ((1 - self.rho) /
                (1 - self.rhohat))).reshape(-1, 1)
            del2 *= self.a2 * (1 - self.a2)

            gradw2 = np.dot(del3, self.a2.T)
            gradb2 = del3.sum(1).reshape(-1, 1)
            gradw1 = np.dot(del2, X)
            gradb1 = del2.sum(1).reshape(-1, 1)

            self.w1 -= self.alpha * (gradw1 / X.shape[0] + self.wdecay * self.w1)
            self.w2 -= self.alpha * (gradw2 / X.shape[0] + self.wdecay * self.w2)
            self.b1 -= self.alpha * (gradb1 / X.shape[0])
            self.b2 -= self.alpha * (gradb2 / X.shape[0])

    def visualize(self, nrows, ncols, show=True, **kwargs):
        w = self.w1.copy()
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
    ae = Autoencoder(layers=[64, 25, 64], alpha=0.3)
    X = get_digits()
    try:
        ae.fit(X, n_iter=1000000)
    except KeyboardInterrupt:
        pass
    ae.visualize(5, 5, interpolation="nearest")
