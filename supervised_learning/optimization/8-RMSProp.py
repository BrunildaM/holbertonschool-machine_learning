#!/usr/bin/env python3
"""
A function that creates the training operation for a neural network
in tensorflow using the RMSProp optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    A function that creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm"""
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
