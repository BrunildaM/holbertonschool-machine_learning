#!/usr/bin/env python3
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    m = Y.shape[1]
    grads = {}
    for i in reversed(range(L)):
        layer = i + 1
        A = cache["A" + str(layer)]
        D = cache["D" + str(layer)]
        if i == 0:
            A_prev = cache["A0"]
        else:
            A_prev = cache["A" + str(i)]
        W = weights["W" + str(layer)]
        b = weights["b" + str(layer)]
        if i == L - 1:
            dZ = A - Y
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        else:
            dA = np.matmul(W.T, dZ)
            dA *= D
            dA /= keep_prob
            dZ = dA * (1 - A ** 2)
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        grads["dW" + str(layer)] = dW
        grads["db" + str(layer)] = db
        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db
