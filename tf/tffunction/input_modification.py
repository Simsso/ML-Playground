from typing import Dict

import tensorflow as tf


@tf.function
def wrong_dict_op(d: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """
    This function modifies the input dictionary.
    TF will raise a ValueError:

    ValueError: dict_op() should not modify its Python input arguments.
    Check if it modifies any lists or dicts passed as arguments. Modifying a copy is allowed.
    """
    a, b = d["a"], d["b"]
    d["c"] = a + b
    return d


@tf.function
def simple_dict_op(d: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """
    This function modifies a copy of its input dict.
    """
    a, b = d["a"], d["b"]
    d_out = d.copy()
    d_out["c"] = a + b
    return d_out


@tf.function
def nested_dict_op(d: Dict[str, Dict[str, tf.Tensor]]) -> Dict[str, Dict[str, tf.Tensor]]:
    """
    This function modifies a dictionary which its input dict references.
    It obfuscates this modification by creating a shallow copy of the input dict.
    """
    inner_d = d["inner"]
    a, b = inner_d["a"], inner_d["b"]

    outer_copy = d.copy()
    outer_copy["inner"]["c"] = a + b

    # the inner dictionary is not affected by the d.copy() operation
    assert outer_copy["inner"] is inner_d

    return outer_copy


def main_flat(dict_op):
    input_spec = {
        "a": tf.TensorSpec(shape=(3,), dtype=tf.int32),
        "b": tf.TensorSpec(shape=(3,), dtype=tf.int32),
    }
    concrete_fn = dict_op.get_concrete_function(input_spec)

    sample_input = {
        "a": tf.convert_to_tensor([1, 2, 3], dtype=tf.int32),
        "b": tf.convert_to_tensor([1, 2, 3], dtype=tf.int32),
    }

    y = concrete_fn(sample_input)

    tf.print(y)


def main_nested(dict_op):
    input_spec = {
        "inner": {
            "a": tf.TensorSpec(shape=(3,), dtype=tf.int32),
            "b": tf.TensorSpec(shape=(3,), dtype=tf.int32),
        },
    }
    concrete_fn = dict_op.get_concrete_function(input_spec)

    sample_input = {
        "inner": {
            "a": tf.convert_to_tensor([1, 2, 3], dtype=tf.int32),
            "b": tf.convert_to_tensor([1, 2, 3], dtype=tf.int32),
        },
    }

    y = concrete_fn(sample_input)

    tf.print(y)


if __name__ == '__main__':
    print("Simple functions")
    flat_fns = (
        wrong_dict_op,
        simple_dict_op,
    )
    for fn in flat_fns:
        print(fn.python_function.__name__)
        try:
            main_flat(fn)
        except Exception as e:
            print(e)

    print("Nested functions")
    nested_fns = (
        nested_dict_op,
    )
    for fn in nested_fns:
        print(fn.python_function.__name__)
        try:
            main_nested(fn)
        except Exception as e:
            print(e)
