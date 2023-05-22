#!/usr/bin/env python3
"""A function  that performs forward propagation
over a convolutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = int(np.ceil((h_prev - 1) / sh))
        pad_w = int(np.ceil((w_prev - 1) / sw))
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0

    out_h = int((h_prev - kh + pad_top + pad_bottom) / sh) + 1
    out_w = int((w_prev - kw + pad_left + pad_right) / sw) + 1

    if padding == "same":
        A_prev = np.pad(A_prev, ((0, 0), (pad_top, pad_bottom),
                                 (pad_left, pad_right), (0, 0)),
                        mode="constant")

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            a_slice_prev = A_prev[:, h_start:h_end, w_start:w_end, :]

            Z[:, i, j, :] = np.sum(a_slice_prev * W, axis=(1, 2, 3))

    Z += b

    A = activation(Z)

    return A
