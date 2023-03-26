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
        public instance method def that calculates PDF at a data point
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.mean.shape[0], 1):
            d = self.mean.shape[0]
            raise ValueError("x must have the shape ({}, 1)".format(d))

        d = self.mean.shape[0]
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        norm_const = 1.0 / (np.power((2*np.pi),
                                     float(d)/2)*np.power(det, 1.0/2))
        x_mu = x - self.mean
        result = np.power(np.e, -0.5*np.dot(np.dot(x_mu.T, inv), x_mu))
        return norm_const*result.flatten()[0]
