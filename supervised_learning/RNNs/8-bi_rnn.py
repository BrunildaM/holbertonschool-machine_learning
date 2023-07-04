#!/usr/bin/env python3
"""
A function that performs forward propagation
for a bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    A function that performs forward propagation
    for a bidirectional RNN
    parameters:
    bi_cell is an instance of BidirectionalCell
    X is the data to be used
      t is the max number of time steps
      m is the batch size
      i is the dimensionality of the data
    h_0 is the initial hidden state in the forward direction
      h is the dimensionality of the hidden state
    h_t is the initial hidden state in the backward direction
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    H_forward[0] = h_0
    H_backward[t - 1] = h_t

    for step in range(t - 1):
        H_forward[step + 1], _ = bi_cell.forward(H_forward[step], X[step])
        H_backward[t - step - 2], _ = bi_cell.backward(
            H_backward[t - step - 1], X[t - step - 1])

    H = np.concatenate((H_forward[:-1], H_backward[1:]), axis=2)
    Y = bi_cell.output(H)

    return H, Y
