import pickle
import sys
from pathlib import Path
import gc

from sklearn.preprocessing import LabelEncoder

sys.path.append("../src")

from src.utils.create_dataset import *
from src.utils.adapt_labels import *


def main(dataset_paths, y_column):
    """
    Creates X_train, X_test, y_train, y_test csvs and stores them in the data_cache folder
    Important: If you want to use fit_cnn.py etc you first need to run this script to create the data_cache

    :param dataset_paths: path of a txt file that contains the paths of the converted audio-labels pairs
    :return:
    """

    # Split dataset by tracks
    logger.info("Splitting dataset...")
    df_train, df_test = split_dataset(dataset_paths, test_size=0.2, random_state=42)

    logger.info("Reading and concatenating train dataset...")
    df_train = read_and_concatenate_files(df_train, dataframe=True, verbose=100)

    logger.info("Reading and concatenating test dataset...")
    df_test = read_and_concatenate_files(df_test, dataframe=True, verbose=100)

    # Extract features and labels
    X_train = df_train.drop('labels', axis=1).values
    X_test = df_test.drop('labels', axis=1).values

    y_train_features = ConvertLab(df_train, label_col='labels', dest=None, is_df=True)
    y_train = y_train_features.df[y_column].values

    y_test_features = ConvertLab(df_test, label_col='labels', dest=None, is_df=True)
    y_test = y_test_features.df[y_column].values

    logger.info(f"Encoding {y_column} labels")
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
    # If folder is non-existent, create it
    Path("data_cache").mkdir(parents=True, exist_ok=True)

    np.savetxt("data_cache/X_train.csv", X_train, delimiter=",")
    np.savetxt("data_cache/X_test.csv", X_test, delimiter=",")
    np.savetxt(f"data_cache/y_train_encoded_{y_column}.csv", y_train_encoded, delimiter=",")
    np.savetxt(f"data_cache/y_test_encoded_{y_column}.csv", y_test_encoded, delimiter=",")

    # Keep labels before encoding too
    df_train['labels'].to_csv("data_cache/y_train.csv", index=False)
    df_test['labels'].to_csv("data_cache/y_test.csv", index=False)

    with open(f'data_cache/label_mapping_{y_column}.pickle', 'wb') as handle:
        pickle.dump(label_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Dataset saved into data_cache folder!")


if __name__ == "__main__":

    gc.collect()

    start = time.time()
    logger.info("Starting up..")

    # dataset_path = '/home/thanos/Documents/Thesis/dataset_paths_transformed.txt'
    dataset_path = '/home/thanos/Documents/Thesis/Dataset_paths/dataset_paths_CQT.txt'
    y_column = 'triad'

    main(dataset_path, y_column)

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
