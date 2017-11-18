import tensorflow as tf
import network


def main(args=None):
    img = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input_batch')
    return network.discriminator(img)


if __name__ == '__main__':
    tf.app.run()