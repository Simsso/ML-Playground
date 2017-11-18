import tensorflow as tf


IMG_WIDTH = 28
IMG_HEIGHT = IMG_WIDTH
INPUT_SIZE = IMG_HEIGHT * IMG_WIDTH
OUTPUT_SIZE = INPUT_SIZE
DENSE1_UNITS = 256
CODE_UNITS = 64
DENSE3_UNITS = DENSE1_UNITS


def encoder(img):
    with tf.variable_scope('encoder'):
        with tf.variable_scope('dense1') as scope:
            weights = tf.get_variable('weights', [INPUT_SIZE, DENSE1_UNITS], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/DENSE1_UNITS))
            biases = tf.get_variable('biases', shape=[DENSE1_UNITS], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            pre_activation = tf.add(tf.matmul(img, weights), biases, name='pre_activation')
            dense1 = tf.sigmoid(pre_activation, name=scope.name)

        with tf.variable_scope('dense2'):
            weights = tf.get_variable('weights', [DENSE1_UNITS, CODE_UNITS], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/CODE_UNITS))
            biases = tf.get_variable('biases', shape=[CODE_UNITS], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            pre_activation = tf.add(tf.matmul(dense1, weights), biases, name='pre_activation')
            code = tf.sigmoid(pre_activation, name='code')

    return code


def decoder(code):
    with tf.variable_scope('decoder'):
        with tf.variable_scope('dense3') as scope:
            weights = tf.get_variable('weights', [CODE_UNITS, DENSE3_UNITS], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/DENSE3_UNITS))
            biases = tf.get_variable('biases', shape=[DENSE3_UNITS], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            pre_activation = tf.add(tf.matmul(code, weights), biases, name='pre_activation')
            dense3 = tf.sigmoid(pre_activation, name=scope.name)

        with tf.variable_scope('dense4'):
            weights = tf.get_variable('weights', [DENSE3_UNITS, OUTPUT_SIZE], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/OUTPUT_SIZE))
            biases = tf.get_variable('biases', shape=[OUTPUT_SIZE], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            pre_activation = tf.add(tf.matmul(dense3, weights), biases, name='pre_activation')
            reconstruction = tf.sigmoid(pre_activation, name='reconstruction')

    return reconstruction


def loss(img, reconstruction):
    loss_array = tf.reduce_mean(tf.pow(tf.subtract(img, reconstruction), 2), [1], name='loss')
    batch_loss = tf.reduce_mean(loss_array, [0], name='batch_loss')
    tf.summary.scalar('loss', batch_loss)
    return loss_array
