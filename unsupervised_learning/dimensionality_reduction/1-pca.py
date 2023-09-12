#!/usr/bin/env python3
"""A function that performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """X is a numpy.ndarray of shape (n, d)
    n: the number of data points
    d: the number of dimensions in each point
    ndim: the new dimensionality of the transformed X
    returns: T a numpy.ndarray of shape (n, ndim)"""
    u, s, vh = np.linalg.svd(X)
    cum = np.cumsum(s)
    thresh = cum[len(cum) - 1] * var
    mask = np.where(thresh > cum)
    var = cum[mask]
    idx = len(var) + 1
    W = vh.T
    Wr = W[:, 0:idx]
    return Wr
