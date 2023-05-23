#!/usr/bin/env python3
"""A function that builds a modified version
of the LeNet-5 architecture using tensorflow"""
import tensorflow as tf

import tensorflow as tf

def lenet5(x, y):
    """A function that builds a modified version of
    the LeNet-5 architecture using tensorflow"""
    initializer = tf.initializers.GlorotUniform()

    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)

    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)

    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    flatten = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(flatten, units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    logits = tf.layers.dense(fc2, units=10, kernel_initializer=initializer)

    y_pred = tf.nn.softmax(logits)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    )

    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return y_pred, train_op, loss, accuracy
