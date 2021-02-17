import click

from complex_model import ComplexModel
from simple_model import SimpleModel

import tensorflow as tf

from string_input_model import StringInputModel


@click.command()
@click.option("--export_path", type=str, help="Path of the folder to which the model weights shall be written.")
def export_for_serving(export_path: str) -> None:
    mock_input = tf.keras.Input(shape=4, batch_size=None, name='main_input', dtype=tf.float32)
    model = SimpleModel()
    model(mock_input)

    # This call fails with
    # Inputs to eager execution function cannot be Keras symbolic tensors
    # suggesting that some transformation is being done by tf.keras.Model.__call__
    #  model.call2(mock_input)

    model.summary()

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None
    )


@click.command()
@click.option("--export_path", type=str, help="Path of the folder to which the model weights shall be written.")
def export_complex_model_for_serving(export_path: str) -> None:
    x_mock = tf.keras.Input(shape=3, name='x', dtype=tf.float32)
    d_mock = {'dict_input1': tf.keras.Input(shape=3, name='d1', dtype=tf.float32)}
    inputs_mock = [x_mock, d_mock]

    model = ComplexModel()
    model(inputs_mock)
    model.summary()

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None
    )


@click.command()
@click.option("--export_path", type=str, help="Path of the folder to which the model weights shall be written.")
def export_string_input_model_for_serving(export_path: str) -> None:
    input_names_mock = tf.convert_to_tensor(['abc', 'def'])
    input_vector_mock = tf.convert_to_tensor([1.3, -10.0, -100.])
    inputs_mock = [input_names_mock, input_vector_mock]

    string_to_int_map = {
        'abc': 0,
        'def': 1,
        'dfa': 2,
    }

    model = StringInputModel(string_to_int_map)
    model(inputs_mock)
    model.summary()

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None
    )


if __name__ == '__main__':
    export_string_input_model_for_serving()
