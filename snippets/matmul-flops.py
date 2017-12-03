import tensorflow as tf
from tensorflow.python.ops.functional_ops import ops as fn_ops

g = tf.Graph()

with g.as_default():
    A = tf.Variable(tf.random_normal([25, 16]))
    B = tf.Variable(tf.random_normal([16, 9]))
    C = tf.matmul(A, B)  # shape=[25,9]

for op in g.get_operations():
    flops = fn_ops.get_stats_for_node_def(g, op.node_def, 'flops').value
    if flops is not None:
        print('Flops should be ~' + str(2*25*16*9))
        print('TF stats gives ' + str(flops))
