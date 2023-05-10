#!/usr/bin/env python3
"""
A function that creates a layer of a neural network using dropout
"""
import numpy as np


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    A function that creates a layer of a neural network usimg droput
    """
    W = np.random.randn(n, prev.shape[0]) * np.sqrt(2 / prev.shape[0])
    b = np.zeros((n, 1))

    Z = np.dot(W, prev) + b

    if activation == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))
    elif activation == 'relu':
        A = np.maximum(0, Z)
    elif activation == 'tanh':
        A = np.tanh(Z)
    else:
        raise ValueError("Invalid activation function.")

    D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
    A *= D
    A /= keep_prob

    return A, D
