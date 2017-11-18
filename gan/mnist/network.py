import tensorflow as tf


def generator(z):
    return None


def discriminator(img):
    """
    :param img: Black-white image tensor 28x28.
    :return: The discriminator output [0,1] and all weights that can be modified.
    """
    weights = []
    with tf.variable_scope('conv1') as scope:
        kernel = _get_wd_variable('kernel', shape=[3, 3, 1, 32], stddev=5e-2, wd=0)
        conv = tf.nn.conv2d(img, kernel, strides=[2, 2, 2, 1], padding='SAME', data_format='NHWC', name='raw_conv')
        bias = _get_variable('bias', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')

        conv1 = tf.nn.relu(pre_activation, name=scope.name)

        weights.append(kernel)
        weights.append(bias)

    with tf.variable_scope('conv2') as scope:
        kernel = _get_wd_variable('kernel', shape=[4, 4, 32, 16], stddev=5e-2, wd=0)
        conv = tf.nn.conv2d(conv1, kernel, strides=[2, 2, 2, 1], padding='SAME', data_format='NHWC', name='raw_conv')
        bias = _get_variable('bias', [16], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, bias, name='pre_activation')
        pool = tf.nn.max_pool(pre_activation, [2, 2, 2, 2], strides=[2, 2, 2, 1], padding='SAME', name='pool')
        conv2 = tf.nn.relu(pool, name=scope.name)

        weights.append(kernel)
        weights.append(bias)

    return conv2, weights


def generator_loss(output, desired):
    return None


def discriminator_loss(output, desired):
    return None


def _get_variable(name, shape, initializer):
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _get_wd_variable(name, shape, stddev, wd):
    dtype = tf.float32
    var = _get_variable(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var