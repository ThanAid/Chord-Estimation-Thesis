"""This script calculates the precision of the predictions for the given data, for n components."""
from src.metrics.mirex import chord_parts_precision
import pandas as pd
from src.utils.load_predictions import load_track
from src.utils.train_utils import get_data_name
from src.post_processing.smoothing import smooth_column, chord_filter
import gc
import time
from loguru import logger

DATA_PATH = "/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/predictions/assembled"
DATA_NAMES = "/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/data_cache_3/df_eval.csv"
SMOOTH: bool = True
FILTER: bool = True
N_PARTS = 1


def main():
    data_paths_df = pd.read_csv(DATA_NAMES)
    preds_total = pd.DataFrame()
    actual_total = pd.DataFrame()
    n_precision_mean = 0
    _n = 0
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

        preds_total = pd.concat([preds_total, preds], axis=0, ignore_index=True)
        actual_total = pd.concat([actual_total, actual], axis=0, ignore_index=True)
        n_precision_mean += chord_parts_precision(reference=actual, prediction=preds, n_parts=N_PARTS)
        _n += 1

    n_precision_mean = n_precision_mean/_n
    n_precision = chord_parts_precision(reference=actual_total, prediction=preds_total, n_parts=N_PARTS)

    print(f"Precision for {N_PARTS} components: {n_precision}")
    print(f"Mean Precision for {N_PARTS} is {n_precision_mean}.")

    return 0


if __name__ == "__main__":
    gc.collect()
    start = time.time()

    main()

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")