import tensorflow as tf
from dataset_reader import MNISTChurner
from network import cnn_layer
from tqdm import tqdm

data_path = "/home/mash/universe/sandbox/mnist_feeder/MNIST_data"

data = MNISTChurner(data_path)
train_data, train_label = data.get_train_data()
print("train data shape is :", train_data.shape)
print("train label shape is :", train_label.shape)

test_data, test_label = data.get_test_data()
print("test data shape is :", test_data.shape)
print("test label shape is :", test_label.shape)

######################### HYPERPARAMETERS #########################
batch_size = 16
num_of_step = 1000
######################### HYPERPARAMETERS #########################

# Graph inputs
x_img = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y_label = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# pass the inputs to the cnn network
y_out = cnn_layer(x_img)

# calculate loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_out)
cross_entropy = tf.reduce_mean(cross_entropy)

# optimize/reduce loss by g-descent
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_reduce = optimizer.minimize(cross_entropy)

# calculate accuracy
pred = tf.argmax(y_out, 1)
correct_class = tf.argmax(y_label, 1)

correct_prediction = tf.equal(pred, correct_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# for tensorboard
#summary_writer = tf.summary.FileWriter('./tf_summary', sess.graph)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for steps in tqdm(range(num_of_step)):

        # TODO create train data batch to train the model
        start = 0
        for batch in tqdm(range(batch_size, train_data.shape[0], batch_size)):
            end = batch
            train_data_batch = train_data[start:end]
            train_label_batch = train_label[start:end]
            #print("training network with {} images".format(train_data_batch.shape[0]))
            start = batch
            sess.run(optimizer_reduce, feed_dict={x_img: train_data_batch, y_label:train_label_batch})

        # TODO test model for the test data
        accuracy_test = sess.run(accuracy, feed_dict={x_img: test_data[:1], y_label:test_label[:1]})
        print(accuracy_test)
