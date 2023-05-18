#!/usr/bin/env python3
"""A function that builds a neural network with the Keras library"""
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """A function that builds a neural network with the Keras
    library"""
    inputs = keras.Input(shape=(nx,))
    x = inputs

    for layer_size, activation_func in zip(layers, activations):
        x = keras.layers.Dense(layer_size, activation=activation_func,
                               kernel_regularizer=keras.regularizers.l2(lambtha))(x)
        x = keras.layers.Dropout(1 - keep_prob)(x)

    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
