#!/usr/bin/env python3
"""a function that converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """a function that converts a one-hot matrix into a vector of labels"""
    try:
        assert isinstance(one_hot, np.ndarray)
        assert len(one_hot.shape) == 2

        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception as e:
        print("Error in one_hot_decode:", str(e))
        return None
