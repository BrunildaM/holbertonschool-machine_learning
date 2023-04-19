#!/usr/bin/env python3
"""
a class that defines a deep neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    a class that defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            if l == 1:
                w = np.random.randn(layers[l-1], nx) * np.sqrt(2/nx)
            else:
                w = np.random.randn(layers[l-1], layers[l-2]) * np.sqrt(2/layers[l-2])
            b = np.zeros((layers[l-1], 1))
            self.__weights['W' + str(l)] = w
            self.__weights['b' + str(l)] = b
