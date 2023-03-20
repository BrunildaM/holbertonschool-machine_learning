#!/usr/bin/env python3
"""A function for Convolution with pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """A function for Convolution with pooling"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pool_h = ((h - kh) // sh) + 1
    pool_w = ((w - kw)  // sw) + 1
    pooled = np.zeros((m, pool_h, pool_w, c))

    i  = 0
    for h in range(0, h - kh + 1, sh):
        j = 0
        for w in range(0, w - kw + 1, sw):
            if mode == 'max':
                output = np.max(images[:, h: h +kh, w: w +kw, :], axis = (1, 2))
                if mode == 'avg':
                    output = np.average(images[:, h: h +kh, w: w +kw, :], axis = (1, 2))
                    pooled[:, i, j, :] = output
                    j += 1
                i += 1
    return pooled
