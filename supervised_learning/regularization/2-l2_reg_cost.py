#!/usr/bin/env python3
"""A function that calculates the cost of a neural network
with L2 regularization"""
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    """
    A function that calculates the cost of a neural network
    with l2 regularization"""
    lambtha = 0.01
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    L2_regularization = tf.reduce_sum([tf.nn.l2_loss(weight) for weight in weights])
    cost = tf.reduce_mean(cost + lambtha * L2_regularization)
    return cost
