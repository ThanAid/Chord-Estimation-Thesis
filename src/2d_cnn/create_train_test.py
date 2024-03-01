import gc
import pickle
import sys
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder

sys.path.append("../src")

from src.utils.create_dataset import *
from src.utils.train_utils import *
from src.utils import label_utils


def main(dataset_paths, y_only, cache_folder='data_cache', lab_column='root', encoding_dict=None):
    """
    Creates X_train, X_test, y_train, y_test csvs and stores them in the data_cache folder
    Important: If you want to use fit_cnn.py etc. you first need to run this script to create the data_cache

    :param y_only: only performs the processing on y column (use if you already have run it once)
    :param lab_column: column to use for labels
    :param dataset_paths: path of a txt file that contains the paths of the converted audio-labels pairs
    :return:
    """

    # Split dataset by tracks
    logger.info("Keeping CD1 and CD2 as evaluation data..")
    dataset_paths_df, df_eval = get_evaluation_set(dataset_paths)

    logger.info("Splitting dataset...")
    df_train, df_test = split_dataset(dataset_paths_df, test_size=0.15, random_state=42, is_df=True)

    logger.info("Init the One hot encoder..")
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.array(list(range(0, len(encoding_dict)))).reshape(-1, 1))

    logger.info("Reading and chunking train dataset...")

    # Initialize data chunking object
    chunker = DataChunking(df_train, dest_file='', chunk_size=100, label_col=lab_column, dataframe=True,
                           encoder=encoder, y_only=y_only, verbose=50, encoding_dict=encoding_dict)

    X_train, y_train = chunker.run_chunkify().get_data()

    y_weights = chunker.get_weights()

    logger.info(f'Shape of train data:\n, {X_train.shape, y_train.shape}')

    logger.info('Saving train data..')

    # If folder is non-existent, create it
    Path(cache_folder).mkdir(parents=True, exist_ok=True)

    if not y_only:
        with open(f'{cache_folder}/X_train.pickle', 'wb') as f:
            pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{cache_folder}/y_train_weights_{lab_column}.pickle', 'wb') as f:
        pickle.dump(y_weights, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{cache_folder}/y_train_{lab_column}.pickle', 'wb') as f:
        pickle.dump(y_train, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Dataset saved into data_cache folder!")

    logger.info("Reading and chunking test dataset...")

    del X_train, y_train, df_train
    gc.collect()

    # Initialize data chunking object
    chunker = DataChunking(df_test, dest_file='', chunk_size=100, label_col=lab_column, dataframe=True,
                           encoder=encoder, y_only=y_only, verbose=50, encoding_dict=encoding_dict)

    X_test, y_test = chunker.run_chunkify().get_data()

    logger.info(f'Shape of test data:\n, {X_test.shape, y_test.shape}')

    logger.info('Saving test data..')

    # If folder is non-existent, create it
    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    
    if not y_only:
        with open(f'{cache_folder}/X_test.pickle', 'wb') as f:
            pickle.dump(X_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'{cache_folder}/y_test_{lab_column}.pickle', 'wb') as f:
        pickle.dump(y_test, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Reading and chunking eval dataset...")

    # Initialize data chunking object
    chunker = DataChunking(df_eval, dest_file='', chunk_size=100, label_col=lab_column, dataframe=True,
                           encoder=encoder, y_only=y_only, verbose=50, encoding_dict=encoding_dict)

    X_eval, y_eval = chunker.run_chunkify().get_data()

    logger.info(f'Shape of eval data:\n, {X_eval.shape, y_eval.shape}')

    logger.info('Saving eval data..')

    # If folder is non-existent, create it
    Path(cache_folder).mkdir(parents=True, exist_ok=True)

    if not y_only:
        with open(f'{cache_folder}/X_eval.pickle', 'wb') as f:
            pickle.dump(X_eval, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{cache_folder}/y_eval_{lab_column}.pickle', 'wb') as f:
        pickle.dump(y_eval, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Dataset saved into {cache_folder} folder!")


if __name__ == "__main__":
    gc.collect()

    start = time.time()
    logger.info("Starting up..")

    DATASET_PATHS = '/home/thanos/Documents/Thesis/Dataset_paths/dataset_paths_CQT.txt'
    CACHE_FOLDER = 'data_cache_3'
    LAB_COLUMN = 'extension_2'
    Y_ONLY = True
    ENCODING_DICT = label_utils.EXT_2_ENCODINGS

    main(DATASET_PATHS, cache_folder=CACHE_FOLDER, y_only=Y_ONLY, lab_column=LAB_COLUMN, encoding_dict=ENCODING_DICT)

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
