# Snippets
## Matmul Flops
A script that determines the flops a matrix multiplication `matmul` takes.

For _A_ with dimensions _ixj_ and _B_ with _jxk_ the number of flops is approximately _2*i*j*k_

The source code is a slightly edited version of https://github.com/tensorflow/tensorflow/issues/899.