#!/usr/bin/env python3
"""A function that builds a neural network with the Keras library"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers


def build_model(nx, layers, activations, lambtha, keep_prob):
    """A function that builds a neural network with the Keras library"""
    model = Sequential()
    model.add(Dense(layers[0], activation=activations[0], input_shape=(nx,),
                    kernel_regularizer=regularizers.l2(lambtha)))
    model.add(Dropout(1 - keep_prob))
    for i in range(1, len(layers)):
        model.add(Dense(layers[i], activation=activations[i],
                        kernel_regularizer=regularizers.l2(lambtha)))
        model.add(Dropout(1 - keep_prob))

    return model
