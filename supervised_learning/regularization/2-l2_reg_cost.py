#!/usr/bin/env python3
"""A function that calculates the cost of a neural network
with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    A function that calculates the cost of a neural network
    with l2 regularization"""
    lambda_param = 0.01
    l2_reg = 0
    for var in tf.trainable_variables():
        if 'weights' in var.name:
            l2_reg += tf.nn.l2_loss(var)
    l2_reg *= lambda_param
    cost += l2_reg
    return cost
