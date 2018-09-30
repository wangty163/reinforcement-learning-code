import tensorflow as tf

def variable_summaries(var, name_prefix):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    tf.summary.scalar('{}_min'.format(name_prefix), tf.reduce_min(var))
    tf.summary.scalar('{}_max'.format(name_prefix), tf.reduce_max(var))
    tf.summary.scalar('{}_mean'.format(name_prefix), mean)
    tf.summary.scalar('{}_stddev'.format(name_prefix), stddev)

def truncated_grads(optimizer, loss, clip_value):
    # get gradients
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    # clip gradients
    cliped_gradients = [None if gradient is None else tf.clip_by_norm(gradient, clip_value) for gradient in gradients]
    # apply cliped gradients
    train_op = optimizer.apply_gradients(zip(cliped_gradients, variables))
    # output for visual in tensorboard
    for grad, var in zip(cliped_gradients, variables):
        if grad is not None:
            tf.summary.histogram(var.name + '/gradient', grad)
    return train_op
