import tensorflow as tf
import ohem.data as data


DENSE1_UNITS = 1024
DENSE2_UNITS = 512


def fnn(x):
    # dense layer 1
    with tf.variable_scope('dense1') as scope:
        weights = tf.get_variable('weights', [data.INPUT_SIZE, DENSE1_UNITS], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/DENSE1_UNITS))
        biases = tf.get_variable('biases', [DENSE1_UNITS], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0, dtype=tf.float32))
        pre_activation = tf.add(tf.matmul(x, weights), biases, name='pre_activation')
        dense1 = tf.sigmoid(pre_activation, scope.name)

    # dense layer 2
    # with tf.variable_scope('dense2') as scope:
    #     weights = tf.get_variable('weights', [DENSE1_UNITS, DENSE2_UNITS], dtype=tf.float32,
    #                               initializer=tf.truncated_normal_initializer(stddev=1.0/DENSE2_UNITS))
    #     biases = tf.get_variable('biases', [DENSE2_UNITS], dtype=tf.float32,
    #                              initializer=tf.constant_initializer(0, dtype=tf.float32))
    #     pre_activation = tf.add(tf.matmul(dense1, weights), biases, name='pre_activation')
    #     dense2 = tf.sigmoid(pre_activation, scope.name)

    # dense layer 3 (output)
    with tf.variable_scope('dense3') as scope:
        weights = tf.get_variable('weights', [DENSE1_UNITS, data.NUM_CLASSES], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/data.NUM_CLASSES))
        biases = tf.get_variable('biases', [data.NUM_CLASSES], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0, dtype=tf.float32))
        dense3_logits = tf.add(tf.matmul(dense1, weights), biases, scope.name)

    # class probabilities
    softmax = tf.nn.softmax(dense3_logits, dim=1, name='softmax')
    return softmax


def l2_loss(prediction, target):
    with tf.variable_scope('loss'):
        mismatch = tf.subtract(prediction, target, name='mismatch')
        l2_norm = 0.5 * tf.reduce_sum(tf.square(mismatch), axis=1, name='l2_norm')

    return l2_norm


def l2_loss_top_k(prediction, target, k):
    target = tf.cast(target, dtype=tf.float32)  # target values come as one hot int32
    with tf.variable_scope('loss_top_k'):
        mismatch = tf.subtract(prediction, target, name='mismatch')
        l2_norm = 0.5 * tf.reduce_sum(tf.square(mismatch), axis=1, name='l2_norm')

        # batch samples with the highest lost
        # top_k returns indices and values; the latter matter here
        top_k_l2_norm = tf.nn.top_k(l2_norm, k=k, sorted=True, name='top_k_l2_norm').values

    return top_k_l2_norm


def accuracy(prediction, target):
    prediction_number = tf.argmax(prediction, axis=1)
    target_number = tf.argmax(target, axis=1)
    acc = tf.metrics.accuracy(target_number, prediction_number, name='accuracy')
    return acc
