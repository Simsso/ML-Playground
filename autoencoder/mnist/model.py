import tensorflow as tf


IMG_WIDTH = 28
IMG_HEIGHT = IMG_WIDTH
INPUT_SIZE = IMG_HEIGHT * IMG_WIDTH
OUTPUT_SIZE = INPUT_SIZE
DENSE1_UNITS = 256
CODE_UNITS = 64
DENSE3_UNITS = DENSE1_UNITS

# Factor by which the penalty -- squared Frobenius of the Jacobian matrix of partial derivaties
# of the code with respect to the inputs -- is weighted.
CONTRACTION_FACTOR = 0.05
WEIGHT_DECAY = 1e-9


def encoder(img):
    with tf.variable_scope('encoder'):
        with tf.variable_scope('dense1') as scope:
            weights = tf.get_variable('weights', [INPUT_SIZE, DENSE1_UNITS], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/DENSE1_UNITS))
            add_weight_decay(weights, WEIGHT_DECAY)
            biases = tf.get_variable('biases', shape=[DENSE1_UNITS], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            add_weight_decay(biases, WEIGHT_DECAY)
            pre_activation = tf.add(tf.matmul(img, weights), biases, name='pre_activation')
            dense1 = tf.sigmoid(pre_activation, name=scope.name)

        with tf.variable_scope('dense2') as scope:
            weights = tf.get_variable('weights', [DENSE1_UNITS, DENSE1_UNITS], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/DENSE1_UNITS))
            add_weight_decay(weights, WEIGHT_DECAY)
            biases = tf.get_variable('biases', shape=[DENSE1_UNITS], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            add_weight_decay(biases, WEIGHT_DECAY)
            pre_activation = tf.add(tf.matmul(dense1, weights), biases, name='pre_activation')
            dense2 = tf.sigmoid(pre_activation, name=scope.name)

        with tf.variable_scope('dense3'):
            weights = tf.get_variable('weights', [DENSE1_UNITS, CODE_UNITS], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/CODE_UNITS))
            add_weight_decay(weights, WEIGHT_DECAY)
            biases = tf.get_variable('biases', shape=[CODE_UNITS], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            add_weight_decay(biases, WEIGHT_DECAY)
            pre_activation = tf.add(tf.matmul(dense2, weights), biases, name='pre_activation')
            code = tf.sigmoid(pre_activation, name='code')

    return code


def decoder(code):
    with tf.variable_scope('decoder'):
        with tf.variable_scope('dense4') as scope:
            weights = tf.get_variable('weights', [CODE_UNITS, DENSE3_UNITS], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/DENSE3_UNITS))
            add_weight_decay(weights, WEIGHT_DECAY)
            biases = tf.get_variable('biases', shape=[DENSE3_UNITS], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            add_weight_decay(biases, WEIGHT_DECAY)
            pre_activation = tf.add(tf.matmul(code, weights), biases, name='pre_activation')
            dense4 = tf.sigmoid(pre_activation, name=scope.name)

        with tf.variable_scope('dense5') as scope:
            weights = tf.get_variable('weights', [DENSE3_UNITS, DENSE3_UNITS], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0 / DENSE3_UNITS))
            add_weight_decay(weights, WEIGHT_DECAY)
            biases = tf.get_variable('biases', shape=[DENSE3_UNITS], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            add_weight_decay(biases, WEIGHT_DECAY)
            pre_activation = tf.add(tf.matmul(dense4, weights), biases, name='pre_activation')
            dense5 = tf.sigmoid(pre_activation, name=scope.name)

        with tf.variable_scope('dense6'):
            weights = tf.get_variable('weights', [DENSE3_UNITS, OUTPUT_SIZE], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1.0/OUTPUT_SIZE))
            add_weight_decay(weights, WEIGHT_DECAY)
            biases = tf.get_variable('biases', shape=[OUTPUT_SIZE], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            add_weight_decay(biases, WEIGHT_DECAY)
            pre_activation = tf.add(tf.matmul(dense5, weights), biases, name='pre_activation')
            reconstruction = tf.sigmoid(pre_activation, name='reconstruction')

    return reconstruction


def loss(img, code, reconstruction):
    # standard reconstruction loss
    loss_array = tf.reduce_mean(tf.pow(tf.subtract(img, reconstruction), 2), [1], name='loss_array')
    batch_loss = tf.reduce_mean(loss_array, [0], name='batch_loss')
    tf.summary.scalar('loss', batch_loss)
    tf.add_to_collection('loss', tf.multiply(batch_loss, (1-CONTRACTION_FACTOR)))

    # contraction penalty
    jacobian = tf.gradients(code, img, name='jacobian')
    penalty_array = tf.square(tf.norm(jacobian, ord='euclidean', axis=[1, 2]), name='penalty_array')
    penalty = tf.reduce_mean(penalty_array)
    tf.summary.scalar('contractive_penalty', penalty)
    tf.add_to_collection('loss', tf.multiply(penalty, CONTRACTION_FACTOR))

    loss_terms = tf.get_collection('loss')
    return tf.add_n(loss_terms, name='total_loss')


def anomaly_map(img, reconstruction):
    delta = tf.subtract(img, reconstruction, name='delta')
    normalized = tf.divide(tf.abs(delta), 2.0, name='normalized_delta')
    return normalized


def add_weight_decay(var, decay_factor):
    weight_decay = tf.multiply(tf.nn.l2_loss(var), decay_factor)
    tf.add_to_collection('loss', weight_decay)
