import pickle
import sys
from pathlib import Path
import gc

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

sys.path.append("../src")

from src.utils.create_dataset import *
from src.utils.train_utils import *


def main(dataset_paths):
    """
    Creates X_train, X_test, y_train, y_test csvs and stores them in the data_cache folder
    Important: If you want to use fit_cnn.py etc you first need to run this script to create the data_cache

    :param dataset_paths: path of a txt file that contains the paths of the converted audio-labels pairs
    :return:
    """

    # Split dataset by tracks
    logger.info("Splitting dataset...")
    df_train, df_test = split_dataset(dataset_paths, test_size=0.2, random_state=42)

    logger.info("Init the One hot encoder..")
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.array(list(range(0, 13))).reshape(-1, 1))

    logger.info("Reading and chunking train dataset...")

    X_train, y_train = (DataChunking(df_train, dest_file='', chunk_size=100, label_col='root', dataframe=True,
                                     encoder=encoder, y_only=True, verbose=50)
                        .run_chunkify().get_data())

    logger.info(f'Shape of train data:\n, {X_train.shape, y_train.shape}')

    logger.info('Saving train data..')

    # If folder is non-existent, create it
    Path("data_cache").mkdir(parents=True, exist_ok=True)

    # with open('data_cache/X_train.pickle', 'wb') as f:
        # pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data_cache/y_train_root.pickle', 'wb') as f:
        pickle.dump(y_train, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Dataset saved into data_cache folder!")

    logger.info("Reading and chunking test dataset...")

    del X_train, y_train, df_train
    gc.collect()

    X_test, y_test = (DataChunking(df_test, dest_file='', chunk_size=100, label_col='root', dataframe=True,
                                   encoder=encoder, y_only=True, verbose=50)
                      .run_chunkify().get_data())

    logger.info(f'Shape of test data:\n, {X_test.shape, y_test.shape}')

    logger.info('Saving test data..')

    # If folder is non-existent, create it
    Path("data_cache").mkdir(parents=True, exist_ok=True)

    # with open('data_cache/X_test.pickle', 'wb') as f:
    #     pickle.dump(X_test, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data_cache/y_test_root.pickle', 'wb') as f:
        pickle.dump(y_test, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Dataset saved into data_cache folder!")


if __name__ == "__main__":
    gc.collect()

    start = time.time()
    logger.info("Starting up..")

    DATASET_PATHS = '/home/thanos/Documents/Thesis/Dataset_paths/dataset_paths_CQT.txt'

    main(DATASET_PATHS)

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
