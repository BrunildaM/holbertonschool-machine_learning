#!/usr/bin/env python3
"""A function that calculates the cost of a neural network
with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    A function that calculates the cost of a neural network
    with l2 regularization"""
    lambtha = 0.01
    weights = [v for v in tf.trainable_variables() if 'kernel' in v.name]
    L2_regularization = tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
    regularized_cost = cost + lambtha * L2_regularization
    return regularized_cost
