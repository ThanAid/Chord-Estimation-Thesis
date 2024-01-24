from tensorflow.keras.utils import Sequence
import numpy as np


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
