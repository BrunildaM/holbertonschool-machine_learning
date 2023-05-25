#!/usr/bin/env python3
"""A function that builds a modified version
of the LeNet-5 architecture using tensorflow"""
import tensorflow as tf


def lenet5(x, y):
    """A function that builds a modified version of the
    LeNet-5 architecture using tensorflow"""
    conv1 = tf.keras.layers.Conv2D(
        filters=6, kernel_size=5, padding='same',
        activation='relu',
        kernel_initializer='he_normal')(x)

    pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=2, strides=2)(conv1)

    conv2 = tf.keras.layers.Conv2D(
        filters=16, kernel_size=5, padding='valid',
        activation='relu',
        kernel_initializer='he_normal')(pool1)

    pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=2, strides=2)(conv2)

    flatten = tf.keras.layers.Flatten()(pool2)

    fc1 = tf.keras.layers.Dense(
        120, activation='relu',
        kernel_initializer='he_normal')(flatten)

    fc2 = tf.keras.layers.Dense(
        84, activation='relu',
        kernel_initializer='he_normal')(fc1)

    output = tf.keras.layers.Dense(
        10, activation='softmax')(fc2)

    loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(y, output))

    accuracy = tf.reduce_mean(
        tf.keras.metrics.categorical_accuracy(y, output))

    optimizer = tf.keras.optimizers.Adam()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accuracy
