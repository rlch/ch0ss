import logo
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

"""
TODO:

    Cross-validation - eh my computer is kinda stinky
    Maybe add shards if the numpy shuffling + splitting is too slow
        - Shuffle each shard
    Actually structure the training
    
"""


def load_dataset(location, split=0.2):
    """

    """
    with np.load(location) as data:
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        return train_data, test_data


def build():
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(loss='mean_absolute_error',
                  optimizer=optimizers.Adam(0.001))

    return model


if __name__ == '__main__':
    logo.show()
    print('\n\n')

    if len(sys.argv) == 1 or not os.path.isfile(sys.argv[1]):
        print('Dataset file location not found. Pass as an argument:')
        print('\t> python train.py ./dataset/processed/dataset.npz\n')
    elif sys.argv[1][-4:] != '.npz':
        print('Dataset must be an npz file. See parse_dataset.py')
    else:
        X_train, X_test, y_train, y_test = load_dataset(sys.argv[1])
        print(len(X_train), len(X_test), len(y_train), len(y_test))
