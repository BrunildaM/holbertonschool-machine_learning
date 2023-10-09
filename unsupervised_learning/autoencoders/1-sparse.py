#!/usr/bin/env python3
"""
A function that creates a sparse autoencoder
"""
import numpy as np
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    A function that creates a sparse autoencoder
    input_dims: an integer containing the dimensions of the model input
    hidden_layers: a list containing the number of nodes for each hidden
    layer in the encoderm respectively
      the hidden layers should be reversed for the decoder
    latent_dims: an integer containing the dimensions of the latent
    space representation
    lambtha: the regularization parameter used for L1 regularization on
    the encoded output
    returns: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the sparse autoencoder model
    """
    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))
    sparsity =  keras.regularizers.l1(lambtha)
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activity_regularizer=sparsity,
                                 activation='relu')(input_encoder)

    for enc in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[enc],
                                     activity_regularizer=sparsity,
                                     activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)
    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)

    for dec in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[dec],
                                     activation='relu')(decoded)

    last = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    encoder_output = encoder(input_encoder)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
