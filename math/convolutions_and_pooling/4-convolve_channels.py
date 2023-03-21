#!/usr/bin/env python3
"""Convolution with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Convolution with channels"""
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    imp = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    out_h = ((h + 2 * ph - kh) // sh) + 1)
    out_w = ((w + 2 * pw - kw) // sw) + 1)
    out = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output = imp[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel
            out[:, i, j] = np.sum(output, axis=(1, 2, 3))
    return out
