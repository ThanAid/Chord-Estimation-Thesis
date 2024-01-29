import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Generator for batches in TF

    Example of use:

    train_gen = DataGenerator(X_train, y_train, 32)
    test_gen = DataGenerator(X_test, y_test, 32)


    history = model.fit(train_gen,
                        epochs=6,
                        validation_data=test_gen)
    """

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def slice_freeze_model(model, n_sliced_layers=3):
    """
    given a tf keras model and a number of layers to slice (from the end) it returns a sliced model usable for transfer
    learning. If freeze is True it also freezes parameters from conv layers.

    :param model: keras model
    :param n_sliced_layers: (int) number of layers to cut
    :return: keras model
    """

    # remove the last layers (n sclided)
    sliced_model = tf.keras.Sequential(model.layers[:-n_sliced_layers])

    # Freeze, set trainable=False for the layers from loaded_model
    for layer in sliced_model.layers:
        layer.trainable = False

    return sliced_model


def plot_cm(model, X_test, y_test, label_mapping):
    """
    Plots confusion matrix
    :param model: keras model
    :param X_test: array like
    :param y_test: encoded y test
    :param label_mapping: dict with encoding map
    :return:
    """
    y_pred = model.predict(X_test)
    y_pred_id = np.argmax(y_pred, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_id)

    # Get the labels in the correct order
    labels = list(label_mapping.keys())

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
