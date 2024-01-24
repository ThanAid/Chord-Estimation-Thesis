"""
Given a txt file containing the paths you want to include in your dataset
    (format: audio_path_fourier   label_path_transformed)
it concatinates to create the dataset.
"""
import sys

sys.path.append("../src")

from sklearn.model_selection import train_test_split
from src.utils.audio_utils import *
import time
from loguru import logger


def read_and_concatenate_files(file_path, dataframe=False, verbose=0):
    # Read the file into a DataFrame
    if not dataframe:
        df = pd.read_csv(file_path, delimiter=' ', index_col=False, names=['wav', 'labels'], header=None, low_memory=False)
    else:
        df = file_path

    # Create empty DataFrames to store the concatenated data
    # audio_data = pd.DataFrame()
    audio_data = []
    # label_data = pd.DataFrame()
    label_data =[]

    # Iterate through rows and concatenate audio and label data
    for i, (index, row) in enumerate(df.iterrows()):
        audio_path = row['wav']
        label_path = row['labels']

        # Assuming you have a function to read audio and label data, replace the placeholders below
        audio_df = read_transformed_audio(audio_path)
        label_df = pd.read_csv(label_path, header=None, sep=' ')

        # Using lists instead of DataFrames and the concat them at once in the end is way faster
        audio_data.append(audio_df)
        label_data.append(label_df)

        if verbose != 0:
            if i % verbose == 0:
                logger.info(f"{i} iterations completed.")

    # Concat the lists into dataframes
    audio_data_df = pd.concat(audio_data, axis=0, ignore_index=True)
    label_data_df = pd.concat(label_data, axis=0, ignore_index=True)

    # Concatenate the two DataFrames horizontally
    audio_data_df['labels'] = label_data_df[1]

    return audio_data_df


def split_dataset(file_path, test_size=0.2, random_state=42):
    """

    :param file_path:
    :param test_size:
    :param random_state:
    :return:
    """
    df = pd.read_csv(file_path, delimiter=' ', index_col=False, names=['wav', 'labels'], header=None)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    return df_train, df_test


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    dataset = read_and_concatenate_files('/home/thanos/Documents/Thesis/dataset_paths_transformed.txt')

    time_elapsed = time.time() - start
    # logger.info(f"Finished, Found {len(converted_test.df)} chords.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
