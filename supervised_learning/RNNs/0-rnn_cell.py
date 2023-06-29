#!/usr/bin/env python3
"""A class that represents a cell of a simple RNN"""
import numpy as np


class RNNCell:
    """A class that represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """Class constructor
        parameters:
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Wh, Wy, bh, by represent the weights and biases of the cell
        Wh and bh are for the concatenated hidden state and input data
        Wy and by are for the output"""
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """Performs the softmax function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def forward(self, h_prev, x_t):
        """A function that performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i)
        that contains the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
        """
        concatenation = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
