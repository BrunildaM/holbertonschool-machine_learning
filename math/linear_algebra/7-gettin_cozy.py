#!/usr/bin/env python3
"""a function that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """a function that concatenates two matrices along a specific axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        result = mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
    return result
