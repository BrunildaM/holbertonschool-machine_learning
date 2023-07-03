#!/usr/bin/env python3
"""
Defines the class BidirectionalCell that represents a bidirectional RNN cell
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell
    """
    def __init__(self, i, h, o):
        """
        class constructor
        parameters:
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=((2 * h), o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        parameters:
            h_prev: contains previous hidden state
            x_t: contains data input for the cell
                h: dimensionality of hidden state
                m: the batch size for the data
                i: dimensionality of the data
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next
