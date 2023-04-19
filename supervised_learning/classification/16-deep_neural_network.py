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
        for i in layers:
            if not isinstance(i, int) or i < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if i == 0:
                self.__weights["W1"] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights["W{}".format(i+1)] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            self.__weights["b{}".format(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
