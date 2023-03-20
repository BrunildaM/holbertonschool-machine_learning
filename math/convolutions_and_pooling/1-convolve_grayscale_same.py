#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""


import numpy as np
"""Performs a same convolution on grayscale images"""


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, height, width = images.shape
    kh, kw = kernel.shape

    if (kh % 2) == 1 and (kw % 2) == 1:
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        ph = kh // 2
        pw = kw // 2

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant',\
                    constant_values = 0)
    convolted = np.zeros((m, height, width))
    
    for h in range(height - kh + 1):
        for w in range(width - kw + 1):
            out = np.sum(images[:, h: h + kh, w: w + kw]\
                         * kernel, axis = 1).sum(axis = 1)
            convoluted[:, h, w] = output
    return convoluted
