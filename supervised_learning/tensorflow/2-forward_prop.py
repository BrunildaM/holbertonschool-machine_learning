#!/usr/bin/env python3
"""
A function that creates the forward propagation graph for the neural network
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Arguments:
    x -- placeholder for the input data
    layer_sizes list containing the number of nodes in each layer
    of the network
    activations list containing the activation functions for each layer
    of the network

    Returns:
    prediction of the network in tensor form
    """
    prev = x
    for i, size in enumerate(layer_sizes):
        activation = activations[i] if i < len(activations) else None
        name = 'layer_' + str(i + 1)
        prev = create_layer_module(prev, size, activation, name)
    return prev
