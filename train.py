import logo
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

"""
TODO:

    Cross-validation - except my computer is kinda stinky
    Maybe add shards if the numpy shuffling + splitting is too slow
        - Shuffle each shard
    Object-oriented CNN, implement callbacks
    
"""

BATCH_SIZE = 128


def load_dataset(location):
    """
    Loads the dataset from a numpy `.npz` file.

    Args:
        `location`: relative location of the file.
    """

    with np.load(location) as data:
        # shift the channel to index 0 to comply with `channels_last`
        # def shift(x): return np.moveaxis(x, 1, -1)
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']

        train_data = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(BATCH_SIZE)
        test_data = tf.data.Dataset.from_tensor_slices(
            (X_test, y_test)).batch(BATCH_SIZE)

        return train_data, test_data


def build():
    """
    Compiles a CNN to use as a value function.

    Returns:
        `tf.keras.Model` object
    """

    model = models.Sequential()
    model.add(layers.InputLayer((6, 8, 8)))
    # model.add(layers.Conv2D(16, (1, 1), data_format='channels_first'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv2D(64, (1, 1), data_format='channels_first'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv2D(32, (1, 1), data_format='channels_first'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv2D(32, (1, 1), data_format='channels_first'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.1))
    model.add(layers.Flatten(data_format='channels_first'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.005),
                  metrics=['mean_absolute_error'])

    model.summary()
    return model


def train(model: tf.keras.Model, train_data, test_data):
    """
    Trains a compiled Keras model.

    Args:
        `model`:        `tf.keras.Model` to be trained.
        `train_data`:   Training data 
        `test_data`:    Test data

    Returns:
        History + statistics of model fitting process.
    """

    stats = model.fit(train_data,
                      validation_data=test_data,
                      epochs=100)
    return stats


def evaluate(model, stats, test_data):
    """
    Evaluates a fitted Keras model.
    Displays a plot with training + test accuracy over epochs.
    Prints test loss and test accuracy.

    Args:
        `model`:        `tf.keras.Model` to be evaluated.
        `stats`:        History object returned by `model.fit()` to be evaluated.
        `test_data`:    Test data

    Returns:
        History + statistics of model fitting process.
    """

    plt.plot(stats.history['mean_absolute_error'], label='mean_absolute_error')
    plt.plot(stats.history['val_mean_absolute_error'], 
             label='val_mean_absolute_error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean absolute error')
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f'Test loss: {test_loss}, Test MAE: {test_acc}')


if __name__ == '__main__':
    logo.show()
    print('\n\n')

    if len(sys.argv) == 1 or not os.path.isfile(sys.argv[1]):
        print('Dataset file location not found. Pass as an argument:')
        print('\t> python train.py ./dataset/processed/dataset.npz\n')
    elif sys.argv[1][-4:] != '.npz':
        print('Dataset must be an npz file. See parse_dataset.py')
    else:
        train_data, test_data = load_dataset(sys.argv[1])
        model = build()
        stats = train(model, train_data, test_data)
        evaluate(model, stats, test_data)
        model.save(f'./models/model')

        print('Saved model in ./models/model.\nI recommend renaming this to avoid unwanted overwrites.')
        # aka I'm too lazy to give the model a uuid
