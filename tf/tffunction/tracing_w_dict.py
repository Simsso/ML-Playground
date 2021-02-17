import time
from typing import Dict

import tensorflow as tf


def add_one(x: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    tf.print("Print from TF")
    tf.print(x)
    print("Print from Python")
    print(x)
    return {k: tf.nn.relu(v + tf.ones_like(v)) for k, v in x.items()}


def main():
    print("Eager function invocation")
    sample_x = {"a": tf.convert_to_tensor([-4, 0, 4]), "b": tf.convert_to_tensor([-3, -2, 0])}
    sample_y = {"a": tf.convert_to_tensor([-0, 4, 8]), "b": tf.convert_to_tensor([1, 2, 3])}
    add_one(sample_x)
    time.sleep(1)

    print("Function invocation with tf.function")
    function_add_one = tf.function(add_one)
    print("Invocation #1")
    function_add_one(sample_x)
    time.sleep(1)

    print("Invocation #2")
    function_add_one(sample_y)
    time.sleep(1)

    print("Input signature")
    concrete_fn = function_add_one.get_concrete_function(sample_y)
    print(concrete_fn.structured_input_signature)
    graph = concrete_fn.graph
    print("Function graph")
    for node in graph.as_graph_def().node:
        print(f'{node.input} -> {node.name}')
    time.sleep(1)

    print("Invocation #3 (dictionary keys modified)")
    sample_z = {"a": tf.convert_to_tensor([-4, 0, 4])}
    function_add_one(sample_z)
    time.sleep(1)


if __name__ == '__main__':
    main()
