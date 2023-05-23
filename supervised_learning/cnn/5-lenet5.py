#!/usr/bin/env python3
"""A function that builds a modified version of the LeNet-5
architecture using keras"""
import tensorflow.keras as K


def lenet5(X):
    """A function that builds a modified version of the
    LeNet-5 achitecture using keras"""
    initializer = K.initializers.he_normal()
    
    conv1 = K.layers.Conv2D(
        6, kernel_size=(5, 5), padding='same',
        activation='relu',
        kernel_initializer=initializer)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(16, kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer=initializer)(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(120, activation='relu',
                         kernel_initializer=initializer)(flatten)

    fc2 = K.layers.Dense(84, activation='relu',
                         kernel_initializer=initializer)(fc1)

    output = K.layers.Dense(10, activation='softmax')(fc2)

    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
