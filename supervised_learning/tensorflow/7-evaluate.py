#!/usr/bin/env python3
"""
A function that evaluates the output of a neural network
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    A function that evaluates the outout of a neural network
    """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(save_path + '.meta')
        loader.restore(sess, save_path)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        y_pred = graph.get_tensor_by_name('y_pred:0')
        loss = graph.get_tensor_by_name('loss:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')

        y_pred, loss, accuracy = sess.run([y_pred, loss, accuracy],
                                          feed_dict={x: X, y: Y})

    return y_pred, accuracy, loss
