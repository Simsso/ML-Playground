import time

import tensorflow as tf


def add_one(x: tf.Tensor) -> tf.Tensor:
    tf.print("Print from TF")
    tf.print(x)
    print("Print from Python")
    print(x)
    return tf.nn.relu(x + tf.ones_like(x))


def main():
    print("Eager function invocation")
    sample_x = tf.convert_to_tensor([-4, 0, 4])
    add_one(sample_x)
    time.sleep(1)

    print("Function invocation with tf.function")
    function_add_one = tf.function(add_one)
    print("Invocation #1")
    function_add_one(sample_x)
    time.sleep(1)

    print("Invocation #2")
    function_add_one(sample_x)
    time.sleep(1)


if __name__ == '__main__':
    main()
