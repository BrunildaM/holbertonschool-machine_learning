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
