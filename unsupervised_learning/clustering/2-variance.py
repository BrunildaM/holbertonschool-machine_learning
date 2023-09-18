#!/usr/bin/env python3
"""A function that calculates the total intra-cluster variance
for a data set"""
import numpy as np


def variance(X, C):
    """A funtion that calcultes the total intra-cluster
    variance for a data set
    X: is a numpy.ndarray of shape (n, d) containing the data set
    C: is a numpy.ndarray of shape (k, d) containing the
    centroid means for each cluster
    returns: var or None on failure
    var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(X.shape) != 2:
        return None
    k, d = C.shape
    if type(k) is not int or k <= 0:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    D = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
    cluster = np.min(D, axis=0)
    var = np.sum(np.square(cluster))
    return var