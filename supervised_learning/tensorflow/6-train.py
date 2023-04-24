#!/usr/bin/env python3
"""
A function  that builds, trains, and saves a neural network classifier
"""
import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier"""
    tf.reset_default_graph()
    m, nx = X_train.shape
    classes = Y_train.shape[1]
    x, y = create_placeholders(nx, classes)
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
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(iterations + 1):
            cost_train, accuracy_train = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            cost_valid, accuracy_valid = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(accuracy_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(accuracy_valid))
            if i != iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        save_path = saver.save(sess, save_path)
        print("Model saved in path: {}".format(save_path))
    return save_path
