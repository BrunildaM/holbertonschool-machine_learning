#!/usr/bin/env python3
"""
A function  that builds, trains, and saves a neural network classifier
"""
import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier
    """
    tf.set_random_seed(0)
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations + 1):
            train_cost, train_acc = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations or i == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))
            if i != iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        save_path = tf.train.Saver().save(sess, save_path)
    return save_path
