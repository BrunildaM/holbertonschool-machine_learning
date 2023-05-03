#!/usr/bin/env python3
"""
A function that creates the training operation for a neural network
in tensorflow using the Adam optimization algorithm
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    A function that creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon)
    return optimizer.minimize(loss)