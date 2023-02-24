#!/usr/bin/env python3
import numpy as np
"""function that concatenates two matrices along a specific axis"""


def np_cat(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specific axis"""
    mat = np.concatenate((mat1, mat2), axis)
    return mat
mat1 = np.array([[11, 22, 33], [44, 55, 66]])
mat2 = np.array([[1, 2, 3], [4, 5, 6]])
mat3 = np.array([[7], [8]])
print(np_cat(mat1, mat2))
print(np_cat(mat1, mat2, axis=1))
print(np_cat(mat1, mat3, axis=1))
