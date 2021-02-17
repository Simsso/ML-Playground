import tensorflow as tf


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.vector_out = tf.keras.layers.Dense(2, activation=None, use_bias=True)
        #  self.scalar_out = tf.keras.layers.Dense(1, activation=None, use_bias=True)

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.vector_out(x)

    @tf.function
    def call2(self, x: tf.Tensor) -> tf.Tensor:
        return self.vector_out(x) + 100

    def call3(self, x: tf.Tensor) -> tf.Tensor:
        return self.vector_out(x) - 100
