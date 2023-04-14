#!/usr/bin/env python3
"""
a class that defines a deep neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")
        
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        for l in range(1, self.L + 1):
            if l == 1:
                self.weights[f"W{l}"] = np.random.randn(layers[l-1], nx) * np.sqrt(2/nx)
            else:
                self.weights[f"W{l}"] = np.random.randn(layers[l-1], layers[l-2]) * np.sqrt(2/layers[l-2])
            self.weights[f"b{l}"] = np.zeros((layers[l-1], 1))
