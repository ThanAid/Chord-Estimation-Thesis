import pickle
import sys

from sklearn.preprocessing import LabelEncoder

sys.path.append("../src")

from src.utils.create_dataset import *
from src.adapt_labels import *


def main(dataset_paths):
    """

    :param dataset_paths:
    :return:
    """

    # Split dataset by tracks
    logger.info("Splitting dataset...")
    # Only keep 50% of dataset
    _, df = split_dataset(dataset_paths, test_size=0.5, random_state=42)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
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

    logger.info("Saving Dataset...")
    np.savetxt("data_cache/X_train_debug.csv", X_train, delimiter=",")
    np.savetxt("data_cache/X_test_debug.csv", X_test, delimiter=",")
    np.savetxt("data_cache/y_train_encoded_debug.csv", y_train_encoded, delimiter=",")
    np.savetxt("data_cache/y_test_encoded_debug.csv", y_test_encoded, delimiter=",")

    with open('data_cache/label_mapping_debug.pickle', 'wb') as handle:
        pickle.dump(label_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Dataset saved into data_cache folder!")


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    # dataset_path = '/home/thanos/Documents/Thesis/dataset_paths_transformed.txt'
    dataset_path = '/home/thanos/Documents/Thesis/Dataset_paths/dataset_paths_CQT.txt'

    main(dataset_path)

    time_elapsed = time.time() - start
    # logger.info(f"Finished, Found {len(converted_test.df)} chords.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
