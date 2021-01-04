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
    Uses a tanh activation function for the final layer to restrict output to [-1, 1].

    Returns:
        `tf.keras.Model` object
    """

    model = models.Sequential()
    model.add(layers.Conv2D(16, (2, 2), input_shape=(
        BATCH_SIZE, 6, 8, 8), data_format='channels_first'))
    model.add(layers.Conv2D(32, (2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='tanh'))
    # tanh gives output in [-1, 1].

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(0.001),
                  metrics=['accuracy'])
    return model


def train(model, train_data, test_data):
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

    plt.plot(stats.history['accuracy'], label='accuracy')
    plt.plot(stats.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')


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
