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
    layer_sizes -- list containing the number of nodes in each layer of the network
    activations -- list containing the activation functions for each layer of the network

    Returns:
    prediction of the network in tensor form
    """
    A = x

    for i in range(len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])
        A = tf.identity(A, name='layer{}'.format(i+1))

    return A
