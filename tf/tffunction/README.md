# tf.function

By default TF runs in eager mode, what does that mean?

```python
def function_to_get_faster(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

y = function_t(t1, t2, t3)
```

The actual Python function gets called.
The values know how they were constructed. That's why, without graph it is still possible to compute derivatives.

tf.function (https://www.tensorflow.org/api_docs/python/tf/function)

```
wrapper = tf.function(fn)
```

The wrapper is doing fancy things:

* Wrapper has the same signature
* Upon calling, the wrapper creates a concrete function
* Next time calling with the same data types and shapes, the concrete function gets reused and Python is not used anymore

converts functions into a graph (distinguish Function, ConcreteFunction, Graph)

What is tracing?

When does tracing happen: (1) first call with specific hash of (type, shape) tuples, (2) explicity tracing with `Function.get_concrete_function`, (3) specify the `input_signature` upfront (?)

```python
@tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.int32)])
def py_function(x):
    return x + 1
```

```python
def my_function(x):
  print("Running eagerly or tracing!")
  if tf.reduce_sum(x) <= 1:
    return x * x
  else:
    return x-1

a_function = tf.function(my_function)

a_concrete_function = a_function.get_concrete_function(input_signature)
y = a_concrete_function(x)
```

Pitfalls: calling with Python primitives (see below) retraces every time, first tracing, debugging (`tf.print` or `run_functions_eagerly`)

```python
@tf.function
def py_function(x, y: int):
    print(x)
    tf.print(x)
    return x + y
```

```python
tf.config.run_functions_eagerly(True)
```

What is a graph, which speedups does it bring? How did that work in TF1?

Advantages: parallelization, constant inlining, no more Python

```python
graph = a_conrete_function.graph
for node in graph.as_graph_def().node:
    print(f'{node.input} -> {node.name}')
```

TF1 (always specify the graph):

```python
x = tf.placeholder(shape=(None, 5))
y = tf.matmul(x, x)
```

Python args in traced functions:

> Prior to TensorFlow 2.3, Python arguments (primitive types) were simply removed from the concrete function's signature. Starting with TensorFlow 2.3, Python arguments remain in the signature, but are constrained to take the value set during tracing.

## Resources

* `tf.function` docs: https://www.tensorflow.org/api_docs/python/tf/function
* Intro: https://www.tensorflow.org/guide/intro_to_graphs
* Details and tricks: https://www.tensorflow.org/guide/function

## Scripts

This folder contains a number of scripts that experiment with tracing.
This section describes the scripts.

### Tracing

**`tracing_simple.py`**

* Normal function invocations are eager
* After the application of `tf.function(fn)` the first invocation traces and the following ones are non-eager.

**`tracing_w_primitive.py`**

* Retracing with a primitive value happens any time the primitive changes.

**`tracing_w_dict.py`**

* A dictionary is not treated any different than a list of parameters.
* If keys or shapes change the function is retraced. 

### Input Modification

The script **`input_modification.py`** analyzes the TF behavior of `tf.function`s which modify their Python inputs.
That is for example adding an element to a list or extending an input dictionary.

TensorFlow detects such behavior and raises a `ValueError`:

> ValueError: dict_op() should not modify its Python input arguments.
> Check if it modifies any lists or dicts passed as arguments. Modifying a copy is allowed.

In tf.data pipelines I did not see this behavior in our production code.
Still, it seems to be a good practice to create copies of input lists or dictionaries (nested copies, if needed).
