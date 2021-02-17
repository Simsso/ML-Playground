# TF Serving Experiments

## Exporting for Serving

The script `export_for_serving.py` exports a simple model (`SimpleModel` class) to a folder containing a `.pb` file.
This format is needed for TF Serving.
Note that the model's batch size is not defined, it is _variable_.

Run the export script
```bash
python tfserving-experiments/export_for_serving.py --export_path=<export-path>/model-name
```

Once the script has written the data to the folder, running

```bash
saved_model_cli show --dir <export-path>/model-name --all
```

reveals from the log output, that only the functions get exported which are annotated with `tf.function`:

```
Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 20), dtype=tf.float32, name='input_1')

  Function Name: '_default_save_signature'
    Option #1
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 20), dtype=tf.float32, name='input_1')

  Function Name: 'call'
    Option #1
      Callable with:
        Argument #1
          x: TensorSpec(shape=(None, 20), dtype=tf.float32, name='x')

  Function Name: 'call2'

  Function Name: 'call_and_return_all_conditional_losses'
    Option #1
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 20), dtype=tf.float32, name='input_1')
```

## Run TF Serving Server

(from this guide https://www.tensorflow.org/tfx/serving/docker)

```
docker run -t --rm -p 8501:8501 \
    -v "<export-path>/model-name:/models/model-name/1" \
    -e MODEL_NAME=model-name \
    tensorflow/serving
```

Without `/1` after the mount directory the serving does not seem to work.

The REST API can be queried with curl:
```
$ curl -d '{"instances": [[1.0, 2.0, 5.0, 4.1], [1.0, 2.0, 5.0, 4.1], [1.0, 2.0, 5.0, 4.1], [1.0, 2.0, 5.0, 4.1]]}' \
    -X POST http://localhost:8501/v1/models/model-name:predict
{
    "predictions": [
        [-3.70516586, -3.57277417, -0.623219848, -0.944630086],
        [-3.70516586, -3.57277417, -0.623219848, -0.944630086],
        [-3.70516586, -3.57277417, -0.623219848, -0.944630086],
        [-3.70516586, -3.57277417, -0.623219848, -0.944630086]
    ]
}
```

Or metadata with

```
curl -X GET http://localhost:8501/v1/models/model-name/metadata
```

## Complex Model Serving

The complex model (class `ComplexModel`) has different input parameters (a tensor and a dict of tensors).

After export, serving is started with

```bash
docker run -t --rm -p 8501:8501
    -v "/Users/tdenk/tmp/complex_model:/models/complex_model/1" \
    -e MODEL_NAME=complex_model \
    tensorflow/serving
```

The metadata reveals that all nested input parameters can be passed

```
curl -X GET http://localhost:8501/v1/models/complex_model/metadata
```

a sample **model call** is

```
curl -d '{"inputs": {"input_1": [[1.0, 2.0, 4.0], [5.0, 4.1, -1.0]], "input_2_dict_input1": [[1.0, 2.0, 4.0], [5.0, 4.1, -1.0]]}}' \
    -X POST http://localhost:8501/v1/models/complex_model:predict
```

Note that the input names to not match the parameter names (in fact the function has only one parameter called `inputs`).
The names are taken from the metadata output. More desirable would be a nested, original structure. However, this does
not seem to be possible. Instead, `"inputs"` vs. `"instances"` differ in the list of dicts or dict of lists structure.
Something as shown below is not possible (as it seems):

```
curl -d '{"instances": [{"inputs": [[[1.0, 2.0, 4.0], [5.0, 4.1, -1.0]], {"dict_input1": [[1.0, 2.0, 4.0], [5.0, 4.1, -1.0]]}]}]}' \
    -X POST http://localhost:8501/v1/models/complex_model:predict
```

{
    "instances": [
        {
            "inputs": [
                [[1.0, 2.0, 4.0], [5.0, 4.1, -1.0]],
                {
                    "dict_input1": [[1.0, 2.0, 4.0], [5.0, 4.1, -1.0]]
                }
            ]
        }
    ]
}


The correct call in `"instances"` mode is:
```
curl -d '{"instances": [{"input_1": [1.0, 2.0, 4.0], "input_2_dict_input1": [5.0, 4.1, -1.0]}]}' \
    -X POST http://localhost:8501/v1/models/complex_model:predict
```

It seems like previous calls were actually having a batch size of two. For more batches with _instances_ one would
expand the instances list by new objects.

This explanation in the TF docs helped here:

```
{
  // (Optional) Serving signature to use.
  // If unspecifed default serving signature is used.
  "signature_name": <string>,

  // Input Tensors in row ("instances") or columnar ("inputs") format.
  // A request can have either of them but NOT both.
  "instances": <value>|<(nested)list>|<list-of-objects>
  "inputs": <value>|<(nested)list>|<object>
}
```

## String Inputs

For string inputs, one feeds a `tf.dtypes.string` tensor into the model.
The model – most likely – needs to convert it into something else, for example, an integer.
That can be achieved as follows:

```python
keys = list(str_to_int_map.keys())
vals = [str_to_int_map[k] for k in keys]
keys_tensor = tf.constant(keys, dtype=tf.string)
vals_tensor = tf.constant(vals, dtype=tf.int32)
initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
self.str_to_int = tf.lookup.StaticHashTable(initializer, -1)
```

```python
int_vec = self.str_to_int.lookup(input_names)
```

Sample calls to the model:
```
curl -d '{"instances": [{"input_1": ["abc", "def"], "input_2": [100.0, -100.0, 0.0]}]}' \
    -X POST http://localhost:8501/v1/models/string_mapping_model:predict
```
yielding
```
{
    "predictions": [[99.989006, -99.978569, 0.0112038031], [99.9620667, -100.015648, -0.0115864985]
    ]
}
```

## Relevant Links

* https://www.tensorflow.org/tfx/tutorials/serving/rest_simple
* https://www.tensorflow.org/tfx/serving/api_rest#predict_api
* https://www.tensorflow.org/api_docs/python/tf/keras/Model