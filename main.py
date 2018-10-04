import tensorflow as tf
import numpy as np
from dataset_reader import MNISTChurner
from tensorflow.examples.tutorials.mnist import input_data
from network import cnn_layer

batch_size = 5000
step_size = 5000000

data_dir = "./mnist_data"

# prepare placeholder for inputs
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
x_in = tf.reshape(x, [-1, 28, 28, 1])

# load labels
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# pass input to the network
y_out = cnn_layer(x_in)

# get actual data to be trained
data = input_data.read_data_sets(data_dir, one_hot=True)

# Calculate loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y)
cross_entropy = tf.reduce_mean(cross_entropy)

# Optimize the network by reducing loss
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


summary_writer = tf.summary.FileWriter('./tmp/logs', sess.graph)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    summary = sess.run(merged)

    summary_writer.add_summary(summary)
    # for summaries in summary:
    #     print(summaries)
    #     summary_writer.add_summary(summaries).eval()

    for i in range(step_size):
        batch = data.train.next_batch(batch_size)
        if i % 1000 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
            print("step {}, training accuracy {}".format(i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

        # X = data.test.images.reshape(10, 1000, 784)
        # Y = data.test.labels.reshape(10, 1000, 10)
        #
        # for i in range(10):
        #     test_accuracy = np.mean(sess.run(accuracy, feed_dict={x: X[i], y: Y[i]}))
        #     print("test accuracy: {}".format(test_accuracy))
