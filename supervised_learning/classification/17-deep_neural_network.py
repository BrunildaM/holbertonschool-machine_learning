#!/usr/bin/env python3
"""
Class that defines a deep neural network performing binary classification
"""
import numpy as np

class DeepNeuralNetwork:
    """
    Class that defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and type(x) is int, layers)):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for l in range(1, self.__L + 1):
            self.__weights['W' + str(l)] = np.random.randn(layers[l-1], nx) * np.sqrt(2/nx)
            self.__weights['b' + str(l)] = np.zeros((layers[l-1], 1))
    
    @property
    def L(self):
        return self.__L
    
    @property
    def cache(self):
        return self.__cache
    
    @property
    def weights(self):
        return self.__weights
