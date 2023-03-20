#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""


import numpy as np
"""Performs a same convolution on grayscale images"""


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, height, width = images.shape
    kh, kw = kernel.shape
    out_w = (width - kw + 2) + 1
    out_h = (height - kh + 2) + 1
    convoluted = np.zeros((m, out_h - kh + 1, out_w - kw + 1))
    for h in range(out_h - kh + 1):
        for w in range(out_w - kw + 1):
            out = np.sum(images[:, out_h: out_h + kh, out_w: out_w + kw]\
                         * kernel, axis = 1).sum(axis = 1)
            convoluted[:, out_h, out_w] = out
    return convoluted
