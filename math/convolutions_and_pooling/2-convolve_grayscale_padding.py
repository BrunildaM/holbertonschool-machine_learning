#!/usr/bin/env python3
"""Performs a convolution on grayscale images"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    img_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    convoluted = np.zeros((m, h + 2 * ph - kh + 1, w + 2 * pw - kw + 1))
    for i in range(h + 2 * ph - kh + 1):
        for j in range(w + 2 * pw - kw + 1):
            output = np.sum(img_padded[:, i: i + kh, j: j + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, i, j] = output
    return convoluted
