import tensorflow as tf
from dataset_reader import MNISTChurner
from tensorflow.examples.tutorials.mnist import input_data

data = MNISTChurner("/home/mash/universe/sandbox/mnist_feeder/MNIST_data")
train_data, train_label = data.get_train_data()
print("train data shape is :", train_data.shape)
print("train label shape is :", train_label.shape)

test_data, test_label = data.get_test_data()
print("test data shape is :", test_data.shape)
print("test label shape is :", test_label.shape)
# input = tf.placeholder(shape=(None,))
