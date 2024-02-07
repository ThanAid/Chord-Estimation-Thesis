import pickle
import sys

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam, SGD

sys.path.append("../src")

from src.adapt_labels import *
from src.utils.train_utils import *
from src.utils import label_utils


def main(dataset_file):
    """

    :param dataset_file: path of file containing X_train, X_test, y_train, y_test pickles
    :return:
    """

    with open(f"{dataset_file}/X_train.pickle", 'rb') as f:
        X_train = pickle.load(f)

    with open(f"{dataset_file}/X_test.pickle", 'rb') as f:
        X_test = pickle.load(f)

    with open(f"{dataset_file}/y_train.pickle", 'rb') as f:
        y_train = pickle.load(f)

    with open(f"{dataset_file}/y_test.pickle", 'rb') as f:
        y_test = pickle.load(f)

    with open(f"{dataset_file}/y_train_root.pickle", 'rb') as f:
        y_train_root = pickle.load(f).astype(int)

    with open(f"{dataset_file}/y_test_root.pickle", 'rb') as f:
        y_test_root = pickle.load(f).astype(int)

    with open(f"{dataset_file}/y_train_weights_root.pickle", 'rb') as f:
        y_weights = pickle.load(f)

    # Reshape the features for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    # Using generators to deal with running out of memory on GPU
    train_gen = DataGenerator(X_train, y_train_root, 16)
    test_gen = DataGenerator(X_test, y_test_root, 16)

    # Define the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding="same",
                               input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 3)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(13,
                                                              activation='softmax'))
        # Adjust the number of output units
    ])

    # Print shapes using print statements
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    optimizer = Adam(learning_rate=0.0001)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_gen, epochs=40, validation_data=test_gen, class_weight=y_weights)
    # Save the model if needed
    model.save('models/CQT_cnn_root_40_w.h5')

    plot_cm(model, X_test, y_test_root, label_mapping=label_utils.NOTE_ENCODINGS, is_2d=True)


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    # dataset_path = '/home/thanos/Documents/Thesis/dataset_paths_transformed.txt'
    dataset_file = 'data_cache'

    main(dataset_file)

    time_elapsed = time.time() - start
    # logger.info(f"Finished, Found {len(converted_test.df)} chords.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
