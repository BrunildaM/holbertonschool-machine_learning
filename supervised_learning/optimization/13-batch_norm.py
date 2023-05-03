#!/usr/bin/env python3
"""
A function that normalizes an unactivated output of a neural network
using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    A function that normalizes an unactivated output of a neural
    network using batch normalization"""
    m, n = Z.shape
    mu = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    Z_norm_scaled  = gamma * Z_norm + beta
    return Z_norm_scaled
