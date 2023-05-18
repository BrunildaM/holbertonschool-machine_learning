#!/usr/bin/env python3
"""A function that saves a model's weights"""
import tensorflow.keras as K
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')


def save_weights(network, filename, save_format='h5'):
   """A function that saves a model's weights"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """A function that loads the weights"""
    network.load_weights(filename)
    return None
