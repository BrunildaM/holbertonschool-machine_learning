#!/usr/bin/env python3
"""a function that calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """a function that calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    stds = np.sqrt(np.diag(C))
    corr = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            corr[i, j] = C[i, j] / (stds[i] * stds[j])

    return corr
