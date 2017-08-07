# coding: utf-8
import theano.tensor as T
import numpy as np
from theano import shared
from theano import function
from tqdm import tqdm

text = open("scarlet.txt", "r").read()
chars = set(text)
char2ix = {c: i for i, c in enumerate(chars)}
ix2char = {i: c for c, i in char2ix.items()}

X = np.zeros((len(text), 80))
for i, char in enumerate(text):
    X[i, char2ix[char]] = 1

x, y = T.dvectors("xy")
wxh = shared(np.random.rand(80, 128))
whh = shared(np.random.rand(128, 128))
why = shared(np.random.rand(128, 80))
bh = shared(np.random.rand(128,))
by = shared(np.random.rand(80,))
h = shared(np.random.rand(128,))
l1_op = T.tanh(T.dot(h, whh) + T.dot(x, wxh) + bh)
l2_op = T.dot(l1_op, why) + by
fin_out = T.exp(l2_op) / T.sum(T.exp(l2_op))

loss = T.nnet.categorical_crossentropy(fin_out, y)

gxh, ghh, ghy, gbh, gby = T.grad(loss, [wxh, whh, why, bh, by])
alpha = 0.1

train = function([x, y], loss, updates=[(wxh, wxh - alpha * gxh),
                                        (whh, whh - alpha * ghh),
                                        (why, why - alpha * ghy),
                                        (h, T.tanh(T.dot(h, whh) + T.dot(x, wxh))),
                                        (bh, bh - alpha * gbh),
                                        (by, by - alpha * gby)])
for n_epoch in range(10):
    myloss = 0
    for i in tqdm(range(X.shape[0] - 1)):
        myloss += train(X[i, :], X[i + 1, :])
    print("Loss: {}".format(myloss))
