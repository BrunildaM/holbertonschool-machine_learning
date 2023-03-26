#!/usr/bin/env python3
"""A class that represents a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """A class that represents a Multivariate Normal distribution"""
    def __init__(self, data):
        """Class constructor"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        deviations = data - self.mean
        self.cov = np.dot(deviations, deviations.T) / (data.shape[1] - 1)


    def pdf(self, x):
        """
        public instance method def that calculates the PDF at a data point
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.mean.shape[0], 1):
            d = self.mean.shape[0]
            raise ValueError("x must have the shape ({}, 1)".format(d))

        d = self.mean.shape[0]
        diff = x - self.mean
        exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov)), diff)
        prefactor = 1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov)))
        return prefactor * np.exp(exponent)
