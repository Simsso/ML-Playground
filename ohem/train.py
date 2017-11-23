import tensorflow as tf
import os
from ohem import data, model


STEPS = 30000
TRAIN_BATCH_SIZE = 32 * 8
LEARNING_RATE = 1e-4

"""
Set K either to TRAIN_BATCH_SIZE to train without online hard example mining
or to a value [1,TRAIN_BATCH_SIZE) to pick the top K examples for back propagation
which induce the highest loss.
"""
K = 32

MODEL_NAME = "k" + str(K) + "_lr" + str(LEARNING_RATE) + "_batch" + str(TRAIN_BATCH_SIZE)


def main(args=None):
    img_batch = tf.placeholder(tf.float32, [None, data.INPUT_SIZE], name='img_batch')  # network input batch
    class_batch = tf.placeholder(tf.float32, [None, data.NUM_CLASSES], name='class_batch')  # desired class values
    class_prediction = model.fnn(img_batch)  # predicted by the network

    # loss
    l2_loss = model.l2_loss(class_prediction, class_batch)
    if K < TRAIN_BATCH_SIZE:
        l2_loss = model.l2_loss_top_k(class_prediction, class_batch, K)
    l2_loss_mean = tf.reduce_mean(l2_loss, axis=0, name='l2_loss_mean')
    acc_scalar, acc_op = model.accuracy(class_prediction, class_batch)

    # optimizer and gradients
    sgd = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    grad = sgd.compute_gradients(l2_loss)
    # TODO: compute gradient magnitude and log it to tf.summary
    # from https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/training/optimizer.py#L345
    # vars_and_grad = [v for g, v in grad if g is not None]
    apply_grad_op = sgd.apply_gradients(grad)

    # tensorboard
    tf.summary.scalar('l2_loss_mean', l2_loss_mean)
    tf.summary.scalar('accuracy', acc_scalar)
    summary_merged = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter('log' + os.sep + MODEL_NAME)

    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    for step in range(STEPS):
        img_batch_val, class_batch_val = data.next_train_batch(TRAIN_BATCH_SIZE)
        _, _, summary = sess.run([apply_grad_op, acc_op, summary_merged], feed_dict={
            img_batch: img_batch_val,
            class_batch: class_batch_val
        })

        log_writer.add_summary(summary, step)

if __name__ == '__main__':
    tf.app.run()