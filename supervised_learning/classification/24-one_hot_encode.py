#!/usr/bin/env python3
"""a function that converts a one-hot matrix into a vector of labels"""
import numpy as np

def one_hot_encode(Y, classes):
  """a function that converts a one-hot matrix into a vector of labels"""
    try:
        m = len(Y)
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except:
        return None
