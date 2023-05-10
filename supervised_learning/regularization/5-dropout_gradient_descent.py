#!/usr/bin/env python3
"""
A function  that updates the weights of a neural network with Dropout
regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    A function  that updates the weights of a neural network with Dropout
    regularization using gradient descent
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for l in reversed(range(1, L+1)):
        A_prev = cache['A' + str(l-1)]
        A = cache['A' + str(l)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        if l > 1:
            D = cache['D' + str(l-1)]
            dA = np.dot(W.T, dZ) * D / keep_prob
        else:
            dA = np.dot(W.T, dZ)

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        W -= alpha * dW
        b -= alpha * db

        dZ = dA * (1 - np.power(A, 2))

    weights['W' + str(l)] = W
    weights['b' + str(l)] = b
