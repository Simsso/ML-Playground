import time

import tensorflow as tf


def add_one_and_primitive(x: tf.Tensor, prim: int) -> tf.Tensor:
    tf.print("Print from TF")
    tf.print(x)
    print("Print from Python")
    print(x)
    return tf.nn.relu(x + tf.ones_like(x) + prim)


def main():
    print("Eager function invocation")
    sample_x = tf.convert_to_tensor([-4, 0, 4])
    add_one_and_primitive(sample_x, 1)
    time.sleep(1)

    print("Function invocation with tf.function")
    function_add_one = tf.function(add_one_and_primitive)
    print("Invocation #1")
    function_add_one(sample_x, 1)
    time.sleep(1)

    print("Invocation #2 with same primitive value")
    function_add_one(sample_x, 1)
    time.sleep(1)

    print("Invocation #3 with other primitive value")
    function_add_one(sample_x, 2)
    time.sleep(1)


if __name__ == '__main__':
    main()
