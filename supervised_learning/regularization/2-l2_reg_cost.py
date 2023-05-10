#!/usr/bin/env python3
"""A function that calculates the cost of a neural network
with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    A function that calculates the cost of a neural network
    with l2 regularization"""
    graph = tf.get_default_graph()
    losses = graph.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    cost_reg = cost + tf.add_n(losses)
    return cost_reg
