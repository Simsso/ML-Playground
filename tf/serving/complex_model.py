import tensorflow as tf
from typing import Dict


class ComplexModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.vector_out = tf.keras.layers.Dense(2, activation=None, use_bias=True)

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        x: tf.Tensor = inputs[0]
        d: Dict[str, tf.Tensor] = inputs[1]
        y1 = self.vector_out(x)
        y2 = self.vector_out(d['dict_input1'])
        return y1 + y2
