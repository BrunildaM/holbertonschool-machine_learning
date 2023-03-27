#!/usr/bin/env python3
"""a function that slices a matrix along specific axes"""
import numpy as np


def np_slice(matrix, axes={}):
    """
    a function that slices a matrix along specific axes
    """
    slices = []
    for i in range(matrix.ndim):
        if i in axes:
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None))
    return matrix[tuple(slices)]
