import tensorflow as tf


INPUT_SIZE = 5  # game state is represented as a 5-tuple (from one player's point of view)
DENSE1_UNITS = 20
DENSE2_UNITS = 15
NUM_ACTIONS = 3  # fold, bet, call

# RL hyperparameters
DISCOUNT_FACTOR = .9


def q_fn(x):
    """
    The Q-function assesses all possible actions that can be taken, given a state.
    Two layer feed forward neural network. All layers are fully connected, biases initialized with 0.
    The constants above define the layer sizes.
    :param x: Batch input tensor to the network.
    :return: Action softmax over three values.
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

    with tf.variable_scope('actions') as scope:
        weights = tf.get_variable('weights', [DENSE2_UNITS, NUM_ACTIONS], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1.0 / NUM_ACTIONS))
        biases = tf.get_variable('biases', shape=[NUM_ACTIONS], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        action_q = tf.add(tf.matmul(dense2, weights), biases, name='action_q_value')

    return action_q


def loss(network_q, network_next_q, rewards):
    """
    Computes the loss for a Q-network.
    TODO: Last state in a MDP equals the Q value, so no next_q is necessary / can be computed at all.
    :param network_q: The network's guess for action's Q-values in many situations.
    :param network_next_q: The network's guess for the following action's Q-values.
    :param rewards: The reward that the env gave for taking the actions.
    :return: 0-D tensor with a L2 loss scalar.
    """
    better_q = DISCOUNT_FACTOR * network_next_q + rewards * (1 - DISCOUNT_FACTOR)
    return tf.nn.l2_loss(network_q - better_q)
