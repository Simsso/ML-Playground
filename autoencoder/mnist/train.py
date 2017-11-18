import tensorflow as tf
import numpy as np
import os
from autoencoder.mnist import *

# import MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


LEARNING_RATE = 3e-4
STEPS = 30000
BATCH_SIZE = 512
MODEL_NAME = str(model.CODE_UNITS) + "-code_" + str(BATCH_SIZE) + "-batch_anomaly2"


def main(args=None):
    img_batch = tf.placeholder(tf.float32, shape=[None, model.INPUT_SIZE], name='img_batch')
    code = model.encoder(img_batch)
    reconstruction = model.decoder(code)
    loss = model.loss(img_batch, reconstruction)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    anomaly_map = model.anomaly_map(img_batch, reconstruction)

    tf.summary.image('input', tf.reshape(img_batch, [-1, model.IMG_WIDTH, model.IMG_HEIGHT, 1]), 4)
    tf.summary.image('reconstruction', tf.reshape(reconstruction, [-1, model.IMG_WIDTH, model.IMG_HEIGHT, 1]), 4)
    tf.summary.image('anomalies', tf.reshape(anomaly_map, [-1, model.IMG_WIDTH, model.IMG_HEIGHT, 1]), 4)

    summary_merged = tf.summary.merge_all()  # Tensorboard
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    log_writer = tf.summary.FileWriter('log' + os.sep + MODEL_NAME, sess.graph)

    for step in range(STEPS):
        img_batch_values, _ = mnist.train.next_batch(BATCH_SIZE)
        _, loss_val, summary = sess.run([optimizer, loss, summary_merged], feed_dict={
            img_batch: img_batch_values
        })

        log_writer.add_summary(summary, step)

        if step % 250 == 0:
            # loss_val contains the loss of each element of the batch
            loss_scalar = np.mean(loss_val, axis=0)
            print(str(step) + " \t" + str(loss_scalar))


if __name__ == '__main__':
    tf.app.run()