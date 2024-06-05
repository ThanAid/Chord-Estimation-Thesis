"""This script calculates the accuracy of the predictions for the given data, for EACH track."""
from src.metrics.accuracy import get_accuracy
import pandas as pd
from src.utils.load_predictions import load_track
from src.utils.train_utils import get_data_name
from src.post_processing.smoothing import smooth_column, chord_filter
import gc
import time
from loguru import logger

SAVE_PATH = "results/mean_accuracies_smooth_filter.csv"
DATA_PATH = "/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/predictions/assembled"
DATA_NAMES = "/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/data_cache_3/df_eval.csv"
SMOOTH: bool = True
FILTER: bool = True


def main():
    data_paths_df = pd.read_csv(DATA_NAMES)
    accuracies = pd.DataFrame()
    for i in range(len(data_paths_df)):
        song_name, album_name, artist_name = get_data_name(data_paths_df['wav'][i])
        preds, actual = load_track(artist_name, album_name, song_name, DATA_PATH)
        if SMOOTH:
            for col in preds:
                # smooth all columns
                preds = smooth_column(preds, col, window_size=5, in_place=True)

        if FILTER:
            # use filtering method
            preds = chord_filter(preds)

        accuracies = pd.concat([accuracies, get_accuracy(preds, actual)], axis=0, ignore_index=True)
    accuracies.to_csv(SAVE_PATH)


if __name__ == "__main__":
    gc.collect()
    start = time.time()

    main()

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
