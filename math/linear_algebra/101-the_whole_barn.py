#!/usr/bin/env python3
"""a function that adds two matrices"""


def add_matrices(mat1, mat2):
    """a function that adds two matrices"""
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        if type(mat1[i]) != type(mat2[i]):
            return None
        if isinstance(mat1[i], list):
            if len(mat1[i]) != len(mat2[i]):
                return None
            mat1[i] = add_matrices(mat1[i], mat2[i])
            if mat1[i] is None:
                return None
        else:
            mat1[i] += mat2[i]
    return mat1
                                                                                                                            
