#!/usr/bin/env python3
"""a function that calculates the sum of squared numbers"""


def summation_i_squared(n):
    """a function that calculates the sum of squared numbers"""
    if type(n) is int and n > 0:
        return int((n*(n+1)*(2*n+1))/6)
    return None
