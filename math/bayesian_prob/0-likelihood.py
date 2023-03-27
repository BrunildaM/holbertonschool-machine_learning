#!/usr/bin/env python3
""" a function that calculates hypothetical probabilities"""
import numpy as np


def likelihood(x, n, P):
    """ a function that calculates hypothetical probabilities"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("""
x must be an integer that is greater than or equal to 0""")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    terms = np.arange(1, x+1, dtype=np.float)
    bc = int(np.prod((n - terms + 1) / terms))
    likelihoods = np.array([bc*(p**x)*((1-p)**(n-x)) for p in P])
    
    return likelihoods
