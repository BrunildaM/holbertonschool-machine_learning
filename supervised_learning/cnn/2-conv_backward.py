#!/usr/bin/env python3
"""
A function  that performs back propagation
over a convolutional layer of a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """A function that performs back propagation over a
    convolutional layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    m, h_prev, w_prev, c_prev = A_prev.shape

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            a_slice_prev = A_prev[:, h_start:h_end, w_start:w_end, :]

            for c in range(c_new):
                dA_prev[:, h_start:h_end, w_start:w_end, :] += W[:, :, :, c] *\
                dZ[:, i, j, c]
                dW[:, :, :, c] += np.sum(a_slice_prev *\
                                         dZ[:, i, j, c][:, None, None, None],
                                         axis=0)
                db[:, :, :, c] += np.sum(dZ[:, i, j, c], axis=0)

    if padding == "same":
        pad_h = max((h_prev - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_prev - 1) * sw + kw - w_prev, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        dA_prev = dA_prev[:, pad_top:h_prev - pad_bottom,
                          pad_left:w_prev - pad_right, :]

    return dA_prev, dW, db
