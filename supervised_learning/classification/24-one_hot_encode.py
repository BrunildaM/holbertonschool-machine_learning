#!/usr/bin/env python3
"""a function that converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_encode(Y, classes):
    try:
        assert isinstance(Y, np.ndarray)
        assert isinstance(classes, int)
        assert len(Y.shape) == 1
        
        m = len(Y)
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception as e:
        print("Error in one_hot_encode:", str(e))
        return None
