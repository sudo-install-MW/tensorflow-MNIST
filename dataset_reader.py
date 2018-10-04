import os
import numpy as np
import struct


class MNISTChurner:

    def __init__(self, data_path):
        """
        initializes object with mnist data path
        :param data_path:
        """
        print("Loading dataset path")
        self.train_data_path = os.path.join(data_path, "train-images.idx3-ubyte")
        self.train_label_path = os.path.join(data_path, "train-labels.idx1-ubyte")
        self.test_data_path = os.path.join(data_path, "t10k-images.idx3-ubyte")
        self.test_label_path = os.path.join(data_path, "t10k-labels.idx1-ubyte")
        print("Dataset path loaded successfully")

    def get_train_data(self):

        with open(self.train_data_path, 'rb') as train_img:
            magic, num, rows, cols = struct.unpack(">IIII", train_img.read(16))
            img = np.fromfile(train_img, dtype=np.uint8)
            img_array = img.reshape(num, rows, cols, 1)

        with open(self.train_label_path, 'rb') as train_label:
            magic, num = struct.unpack(">II", train_label.read(8))
            label = np.fromfile(train_label, dtype=np.uint8).reshape(num, 1)
            label_one_hot = []

            # convert label to one hot
            for i in range(label.shape[0]):
                cur_label = label[i]
                label_one_hot.append([1 if i == cur_label else 0 for i in range(10)])

            label = np.asarray(label_one_hot)

        return img_array, label

    def get_test_data(self):

        with open(self.test_data_path, 'rb') as test_img:
            magic, num, rows, cols = struct.unpack(">IIII", test_img.read(16))
            img = np.fromfile(test_img, dtype=np.uint8)
            img_array = img.reshape(num, rows, cols, 1)

        with open(self.test_label_path, 'rb') as test_label:
            magic, num = struct.unpack(">II", test_label.read(8))
            label = np.fromfile(test_label, dtype=np.uint8).reshape(num, 1)

            label_one_hot = []

            for i in range(label.shape[0]):
                cur_label = label[i]
                label_one_hot.append([1 if i == cur_label else 0 for i in range(10)])

            label = np.asarray(label_one_hot)

        return img_array, label
