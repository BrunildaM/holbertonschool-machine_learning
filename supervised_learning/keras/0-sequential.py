#!/usr/bin/env python3
"""A function that builds a neural network with the Keras library"""
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """A function that builds a neural network with the Keras library"""
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layers[0], activation=activations[0],
                                 input_shape=(nx,),
                    kernel_regularizer=keras.regularizers.l2(lambtha)))
    model.add(keras.layers.Dropout(1 - keep_prob))

    for i in range(1, len(layers)):
        model.add(keras.layers.Dense(layers[i], activation=activations[i],
                        kernel_regularizer=keras.regularizers.l2(lambtha)))
        model.add(keras.layers.Dropout(1 - keep_prob))

    return model
