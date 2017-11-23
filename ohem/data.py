# import MNIST data set
from tensorflow.examples.tutorials.mnist import input_data


_mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

IMG_WIDTH = 28
IMG_HEIGHT = IMG_WIDTH
INPUT_SIZE = IMG_HEIGHT * IMG_WIDTH  # number of pixels (grayscale)
NUM_CLASSES = 10  # digits from 0 to 9


def next_train_batch(batch_size):
    return _mnist.train.next_batch(batch_size)