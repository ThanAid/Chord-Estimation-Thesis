import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
import gc

import sys

sys.path.append("../src")

from src.utils.create_dataset import *
from src.utils.adapt_labels import *


def main(dataset_paths):
    """

    :param dataset_paths:
    :return:
    """

    # Split dataset by tracks
    logger.info("Splitting dataset...")
    df_train, df_test = split_dataset(dataset_paths, test_size=0.2, random_state=42)
    logger.info("Reading and concatenating train dataset...")
    df_train = read_and_concatenate_files(df_train, dataframe=True)
    logger.info("Reading and concatenating test dataset...")
    df_test = read_and_concatenate_files(df_test, dataframe=True)

    # Extract features and labels
    X_train = df_train.drop('labels', axis=1).values
    X_test = df_test.drop('labels', axis=1).values
    y_train_features = ConvertLab(df_train, label_col='labels', dest=None, is_df=True)
    y_train = y_train_features.df['root'].values

    y_test_features = ConvertLab(df_test, label_col='labels', dest=None, is_df=True)
    y_test = y_test_features.df['root'].values

    logger.info("Deleting Dataframes from memory..")
    del [[df_train, df_test]]
    gc.collect()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    logger.info("Cleared memory!")

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform all the labels
    y_encoded = label_encoder.fit_transform(np.concatenate((y_test, y_train)))

    # Transform the train and test labels using the same encoder
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Now y_train_encoded and y_test_encoded contain the encoded labels

    # To get the mapping from original labels to encoded labels, you can use:
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Normalize the input features if needed
    # Replace this with your actual normalization logic if required

    # Reshape the features for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

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
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Adjust the number of output units
    ])

    # Print shapes using print statements
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    optimizer = Adam()

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train_encoded, epochs=30, batch_size=64, validation_data=(X_test, y_test_encoded))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Save the model if needed
    model.save('CQT_cnn_root.h5')


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    dataset_path = '/home/thanos/Documents/Thesis/dataset_paths_transformed.txt'
    dataset_path = '/home/thanos/Documents/Thesis/Dataset_paths/dataset_paths_CQT.txt'

    main(dataset_path)

    time_elapsed = time.time() - start
    # logger.info(f"Finished, Found {len(converted_test.df)} chords.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
