import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import time
from loguru import logger
from tensorflow.keras.optimizers import Adam
import gc

import sys
import pickle

sys.path.append("../src")

from src.utils.create_dataset import *
from src.adapt_labels import *
from src.utils.train_utils import *


def main(dataset_file):
    """

    :param dataset_file: path of file containing X_train, X_test, y_train_encoded, y_test_encoded and label_mapping pickle
    :return:
    """

    X_train = np.loadtxt(f"{dataset_file}/X_train.csv", delimiter=",")
    X_test = np.loadtxt(f"{dataset_file}/X_test.csv", delimiter=",")
    y_train_encoded = np.loadtxt(f"{dataset_file}/y_train_encoded.csv", delimiter=",").astype(int)
    y_test_encoded = np.loadtxt(f"{dataset_file}/y_test_encoded.csv", delimiter=",").astype(int)

    with open(f"{dataset_file}/label_mapping.pickle", 'rb') as lm:
        label_mapping = pickle.load(lm)

    # Reshape the features for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Using generators to deal with running out of memory on GPU
    train_gen = DataGenerator(X_train, y_train_encoded, 32)
    test_gen = DataGenerator(X_test, y_test_encoded, 32)

    # Define the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding="same", input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Conv1D(16, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(np.concatenate((y_test_encoded, y_train_encoded)))), activation='softmax')
        # Adjust the number of output units
    ])

    # Print shapes using print statements
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    optimizer = Adam(learning_rate=0.0001)

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_gen, epochs=10, validation_data=test_gen)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Save the model if needed
    model.save('models/CQT_cnn_root_10.h5')


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    # dataset_path = '/home/thanos/Documents/Thesis/dataset_paths_transformed.txt'
    dataset_file = 'data_cache'

    main(dataset_file)

    time_elapsed = time.time() - start
    # logger.info(f"Finished, Found {len(converted_test.df)} chords.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
