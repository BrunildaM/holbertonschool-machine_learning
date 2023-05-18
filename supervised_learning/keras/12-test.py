#!/usr/bin/env python3
"""A function that tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """A function that tests a neural network"""
    loss, accuracy = network.evaluate(x=data,
                                      y=labels,
                                      verbose=verbose)
    return loss, accuracy
