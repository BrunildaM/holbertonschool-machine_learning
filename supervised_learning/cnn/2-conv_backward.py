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
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    if padding == "same":
        pad_h = int(np.ceil((h_prev * (sh - 1) - sh + kh) / 2))
        pad_w = int(np.ceil((w_prev * (sw - 1) - sw + kw) / 2))
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    else:
        A_prev_pad = A_prev

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    dA_prev_pad = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    dW[:, :, :, c] += dA_prev_pad * dZ[i, h, w, c]
                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    if padding == "same":
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]

    return dA_prev, dW, db
