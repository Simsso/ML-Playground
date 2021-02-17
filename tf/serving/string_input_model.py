import tensorflow as tf
from typing import Dict


class StringInputModel(tf.keras.Model):
    """
    This model takes a string input and converts it into an integer.
    That is for example needed for SKU to index conversion.
    An example is here:
    https://github.bus.zalan.do/sprezzatura/optimus-prime/blob/33011930c3f674ae89cabda3d736d1b6f475deca/src/prediction/inference.py#L168
    """
    def __init__(self, str_to_int_map: Dict[str, int], *args, **kwargs):
        super().__init__(*args, **kwargs)

        keys = list(str_to_int_map.keys())
        vals = [str_to_int_map[k] for k in keys]
        keys_tensor = tf.constant(keys, dtype=tf.string)
        vals_tensor = tf.constant(vals, dtype=tf.int32)
        initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        self.str_to_int = tf.lookup.StaticHashTable(initializer, -1)

        self.embedding = tf.keras.layers.Embedding(input_dim=4, output_dim=3)

    @tf.function
    def call(self, inputs, training: bool = False):
        input_names, input_vector = inputs

        # string to int
        int_vec = self.str_to_int.lookup(input_names)

        # int to vec
        mat = self.embedding(int_vec)

        # result computation
        result = tf.reduce_sum(mat, axis=0) + input_vector
        return result
