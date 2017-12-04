import tensorflow as tf


INPUT_SIZE = 5  # game state is represented as a 5-tuple (from one player's point of view)
DENSE1_UNITS = 20
DENSE2_UNITS = 15
OUTPUT_SIZE = 1  # scalar q function


def fnn(x):
    """
    Two layer feed forward neural network. All layers are fully connected, biases initialized with 0.
    The constants above define the layer sizes.
    :param x: Batch input tensor to the network.
    :return: Network output tensor. Sigmoid is the last layer activation, so no softmax is applied yet.
    """
    with tf.variable_scope('dense1') as scope:
        weights = tf.get_variable('weights', [INPUT_SIZE, DENSE1_UNITS], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1.0 / DENSE1_UNITS))
        biases = tf.get_variable('biases', shape=[DENSE1_UNITS], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        pre_activation = tf.add(tf.matmul(x, weights), biases, name='pre_activation')
        dense1 = tf.sigmoid(pre_activation, name=scope.name)

    with tf.variable_scope('dense2') as scope:
        weights = tf.get_variable('weights', [DENSE1_UNITS, DENSE2_UNITS], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1.0 / DENSE2_UNITS))
        biases = tf.get_variable('biases', shape=[DENSE2_UNITS], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        pre_activation = tf.add(tf.matmul(dense1, weights), biases, name='pre_activation')
        dense2 = tf.sigmoid(pre_activation, name=scope.name)

    with tf.variable_scope('dense2') as scope:
        weights = tf.get_variable('weights', [DENSE2_UNITS, OUTPUT_SIZE], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1.0 / OUTPUT_SIZE))
        biases = tf.get_variable('biases', shape=[OUTPUT_SIZE], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        pre_activation = tf.add(tf.matmul(dense2, weights), biases, name='pre_activation')
        output = tf.sigmoid(pre_activation, name=scope.name)

    return output