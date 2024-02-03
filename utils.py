import numpy as np
import tensorflow as tf
import sys
import operator


def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
                   callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if callback and callback_every and num_examples_seen % callback_every == 0:
                callback(model, num_examples_seen)
    return model


def save_model_parameters_tf(model, outfile):
    with tf.compat.v1.Session() as session:
        saver = tf.compat.v1.train.Saver()
        session.run(tf.compat.v1.global_variables_initializer())
        saver.save(session, outfile)
    print("Saved model parameters to %s." % outfile)


def load_model_parameters_tf(path, modelClass):
    with tf.compat.v1.Session() as session:
        saver = tf.compat.v1.train.import_meta_graph(path)
        saver.restore(session, tf.compat.v1.train.latest_checkpoint(path))
        # Assuming modelClass is a class that takes necessary parameters for initialization
        model = modelClass()
        return model


def gradient_check_tf(model, x, y, h=0.001, error_threshold=0.01):
    # Placeholder for inputs
    x_placeholder = tf.compat.v1.placeholder(tf.float32, shape=x.shape)
    y_placeholder = tf.compat.v1.placeholder(tf.float32, shape=y.shape)
    # Build the forward pass of the model
    logits = model.forward_pass(x_placeholder)
    # Loss function
    loss = model.calculate_loss(logits, y_placeholder)
    # Gradient computation
    gradients = tf.gradients(loss, tf.compat.v1.trainable_variables())
    
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        # Compute gradients
        computed_gradients = session.run(gradients, feed_dict={x_placeholder: x, y_placeholder: y})
    
    # Compare computed gradients with backpropagated gradients
    for var, grad in zip(tf.compat.v1.trainable_variables(), computed_gradients):
        var_value = var.eval()
        print("Performing gradient check for parameter %s with size %d." % (var.name, np.prod(var_value.shape)))
        # Iterate over each element of the parameter matrix
        it = np.nditer(var_value, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = var_value[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            var_value[ix] = original_value + h
            var.load(var_value, session)
            loss_plus = session.run(loss, feed_dict={x_placeholder: x, y_placeholder: y})
            var_value[ix] = original_value - h
            var.load(var_value, session)
            loss_minus = session.run(loss, feed_dict={x_placeholder: x, y_placeholder: y})
            estimated_gradient = (loss_plus - loss_minus) / (2 * h)
            var_value[ix] = original_value
            var.load(var_value, session)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = grad[ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is too large, fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check ERROR: parameter=%s ix=%s" % (var.name, ix))
                print("+h Loss: %f" % loss_plus)
                print("-h Loss: %f" % loss_minus)
                print("Estimated_gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative Error: %f" % relative_error)
                return
            it.iternext()
        print("Gradient check for parameter %s passed." % var.name)
