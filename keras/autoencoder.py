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
Modularized keras autoencoder
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


class Autoencoder(object):
    def __init__(self, layers, activations=None, optimizer="adadelta",
            loss="binary_crossentropy", **kwargs):
        self.layers = np.array(layers)
        if activations is None:
            activations = ["relu"] * (len(layers) - 2)
            activations.append("sigmoid")
        self.model_layers = []
        for i_layer, shape in enumerate(layers):
            if i_layer == 0:
                layer = Input(shape=(shape,))
            else:
                layer = Dense(shape, activation=activations[i_layer - 1])(layer)
            self.model_layers.append(layer)
        self.encoded_layer_ix = np.argmin(layers)
        self.encoding_dim = layers[self.encoded_layer_ix]
        self._autoencoder = Model(input=self.model_layers[0],
                output=self.model_layers[-1])
        self._autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, validation_data=None, **kwargs):
        self._autoencoder.fit(x, x, validation_data=validation_data, **kwargs)
        self._encoder = Model(input=self.model_layers[0],
                              output=self.model_layers[self.encoded_layer_ix])
        encoded_input = Input(shape=(self.encoding_dim,))
        decoded_output = encoded_input
        for layer in self._autoencoder.layers[-self.encoded_layer_ix:]:
            decoded_output = layer(decoded_output)
        self._decoder = Model(input=encoded_input, output=decoded_output)

    def predict(self, x, **kwargs):
        return self._autoencoder.predict(x, **kwargs)

    def encode(self, x, **kwargs):
        return self._encoder.predict(x)

    def decode(self, x, **kwargs):
        return self._decoder.predict(x)


def fit_deep_autoencoder(x_train, x_test, encoding_dim=32, **kwargs):
    input_img = Input(shape=(x_train.shape[1],))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(x_train.shape[1], activation='sigmoid')(decoded)

    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
    autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), **kwargs)
    encoder = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoded_output = encoded_input
    for layer in autoencoder.layers[-3:]:
        decoded_output = layer(decoded_output)
    decoder = Model(input=encoded_input, output=decoded_output)
    return encoder, decoder, autoencoder


def visualize(decoded_imgs, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    aenc = Autoencoder(layers=[784, 128, 64, 32, 64, 128, 784])
    aenc.fit(x_train, validation_data=(x_test, x_test), nb_epoch=100,
            batch_size=256, shuffle=True)

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = aenc.encode(x_test)
    print encoded_imgs.mean()
    decoded_imgs = aenc.decode(encoded_imgs)

    visualize(decoded_imgs)
