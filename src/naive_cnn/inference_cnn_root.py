import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import time
from loguru import logger
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


import sys

sys.path.append("../src")

from src.utils.create_dataset import *
from src.adapt_labels import *


def main(dataset_paths):
    """

    :param dataset_paths:
    :return:
    """

    # load the model if needed
    logger.info("Loading the model..")
    model = tf.keras.models.load_model('/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/naive_cnn/naive_cnn_v2.h5')

    # Split dataset by tracks
    df_train, df_test = split_dataset(dataset_paths, test_size=0.2, random_state=42)
    df_train = read_and_concatenate_files(df_train, dataframe=True)
    df_test = read_and_concatenate_files(df_test, dataframe=True)

    # Extract features and labels
    X_train = df_train.drop('labels', axis=1).values
    X_test = df_test.drop('labels', axis=1).values
    y_train_features = ConvertLab(df_train, label_col='labels', dest=None, is_df=True)
    y_train = y_train_features.df['root'].values

    y_test_features = ConvertLab(df_test, label_col='labels', dest=None, is_df=True)
    y_test = y_test_features.df['root'].values

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform all the labels
    y_encoded = label_encoder.fit_transform(np.concatenate((y_test, y_train)))

    # Transform the train and test labels using the same encoder
    # y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Now y_train_encoded and y_test_encoded contain the encoded labels

    # To get the mapping from original labels to encoded labels, you can use:
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Normalize the input features if needed
    # Replace this with your actual normalization logic if required

    # Reshape the features for CNN input
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    y_pred = model.predict(X_test)
    # TODO: Need to apply softmax
    y_pred_id = np.argmax(y_pred, axis=1)


    # Calculate confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred_id)

    # Get the labels in the correct order
    labels = list(label_mapping.keys())

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    print('')


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    dataset_path = '/home/thanos/Documents/Thesis/dataset_paths_transformed.txt'

    main(dataset_path)

    time_elapsed = time.time() - start
    # logger.info(f"Finished, Found {len(converted_test.df)} chords.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
