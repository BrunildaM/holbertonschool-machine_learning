#!/usr/bin/env python3
"""
a class that defines a deep neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    a class that defines a deep neural nework performing binary classification
    """"
    def __init__(self, nx, layers):
        """class conustructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            if l == 1:
                self.__weights["W1"] = np.random.randn(layers[0], nx) * np.sqrt(2/nx)
            else:
                self.__weights[f"W{l}"] = np.random.randn(layers[l-1], layers[l-2]) * np.sqrt(2/layers[l-2])
            self.__weights[f"b{l}"] = np.zeros((layers[l-1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
