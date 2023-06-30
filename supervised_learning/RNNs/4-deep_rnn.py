#!/usr/bin/env python3
"""
Defines function that performs forward propagation for a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    parameters:
    rnn_cells: cells that will be used for the forward propagation
    X: the data to be used
    h_0: the initial hidden state
    """
    layers = len(rnn_cells)
    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0
    for step in range(t):
        for layer in range(layers):
            if layer == 0:
                h_prev = X[step]
            h_prev, y = rnn_cells[layer].forward(H[step, layer], h_prev)
            H[step + 1, layer, ...] = h_prev
            if layer == layers - 1:
                if step == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)
    return (H, Y)   
