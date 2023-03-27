#!/usr/bin/env python3
"""a function that calculates the posterior probability"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    a function that calculates the posterior probability"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(p1, float) or not 0 <= p1 <= 1:
        raise ValueError('p1 must be a float in the range [0, 1]')
    if not isinstance(p2, float) or not 0 <= p2 <= 1:
        raise ValueError('p2 must be a float in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')
    prior = 1
    numerator = special.comb(n, x) * p1 ** x * (1 - p1) ** (n - x)
    denominator = 0
    for p in range(int(p1 * 1000), int(p2 * 1000) + 1):
        p /= 1000.0
        denominator += special.comb(n, x) * p ** x * (1 - p) ** (n - x)
    posterior = numerator / denominator * prior
    return posterior
