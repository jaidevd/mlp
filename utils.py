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
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

STD = 0.35

np.random.seed(42)


def get_weights(shape):
    std = 1.0 / np.sqrt(shape[1] + 1)
    return np.random.normal(scale=std, size=shape)


def get_xor():
    X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
    y = np.array([[1, 0],
                [0, 1],
                [0, 1],
                [1, 0]])
    return X, y


def get_xor_blobs():
    pos_center = np.array([[-1, -1], [1, 1]])
    pos_data = make_blobs(centers=pos_center, cluster_std=STD)[0]
    pos_data = np.c_[pos_data, np.ones((pos_data.shape[0], 1))]

    neg_center = np.array([[1, -1], [-1, 1]])
    neg_data = make_blobs(centers=neg_center, cluster_std=STD)[0]
    neg_data = np.c_[neg_data, np.zeros((neg_data.shape[0], 1))]

    X = np.r_[pos_data, neg_data]
    np.random.shuffle(X)
    y = OneHotEncoder().fit_transform(X[:, 2].reshape(-1, 1))
    X = X[:, :2]
    X = StandardScaler().fit_transform(X)
    return X, y.toarray()


def draw_decision_boundary(bp, X, y, ax=None, show=True, **kwargs):
    y = np.argmax(y, axis=1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = bp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.argmax(axis=1)
    Z = Z.reshape(xx.shape)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(211)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, **kwargs)
    if show:
        plt.show()

if __name__ == '__main__':
    X, y = make_circles(factor=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
