import tensorflow as tf
from tensorflow.python.ops.functional_ops import ops as fn_ops


# matrix multiplication
g = tf.Graph()
with g.as_default():
    a = tf.Variable(tf.random_normal([25, 16]))
    b = tf.Variable(tf.random_normal([16, 9]))
    c = tf.matmul(a, b)  # shape=[25,9]


# convolutional layer
g_conv = tf.Graph()
with g_conv.as_default():
    img = tf.Variable(tf.random_normal([16, 48, 48, 3]))
    kernel = tf.get_variable('weights', shape=[5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
    conv = tf.nn.conv2d(img, kernel, [2, 2, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    out = tf.nn.relu(pre_activation)

print("matmul")
for op in g.get_operations():
    flops = fn_ops.get_stats_for_node_def(g, op.node_def, 'flops').value
    print(op.name + ': TF stats gives ' + str(flops))

print("conv")
for op in g_conv.get_operations():
    flops = fn_ops.get_stats_for_node_def(g_conv, op.node_def, 'flops').value
    print(op.name + ': TF stats gives ' + str(flops))
