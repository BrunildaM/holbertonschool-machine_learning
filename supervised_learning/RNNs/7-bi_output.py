#!/usr/bin/env python3
"""
Represents a bidirectional cell of an RNN
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

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step
        """
        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)

        return h_prev

    def softmax(self, x):
        """
        Performs the softmax function
        x is the value to perrform softmax on to generate output of the cell
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def output(self, H):
        """
        A public instance method that calculates all outputs for the RNN
        parameters:
        H: is a numpy.ndarray
        t is the number of time steps
        m is the batch size for the data
        h is the dimensionality of the hidden states
        """
        t, m, h = H.shape
        Y = []
        for step in range(t):
            y = self.softmax(np.matmul(H[step], self.Wy) + self.by)
            Y.append(y)
        return np.array(Y)
