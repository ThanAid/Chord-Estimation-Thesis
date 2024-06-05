"""Calculate the resuts using the MIREX evaluation metrics.

Those are:
-Chord root note only;
-Major and minor: {N, maj, min};
-Seventh chords: {N, maj, min, maj7, min7, 7};
-Major and minor inversions: {N, maj, min, maj/3, min/b3, maj/5, min/5};
-Seventh chords with inversions: {N, maj, min, maj7, min7, 7, maj/3, min/b3, maj7/3, min7/b3, 7/3, maj/5, min/5,
 maj7/5, min7/5, 7/5, maj7/7, min7/b7, 7/b7};
"""

import gc
import time
import warnings

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.metrics.mirex import majmin_accuracy, seventh_accuracy, maj_inv_accuracy, csr_accuracy, seventh_inv_acccuracy, mirex_accuracy
from src.post_processing.smoothing import smooth_column, chord_filter
from src.utils.load_predictions import load_track
from src.utils.train_utils import get_data_name

warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_PATH = "/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/predictions/assembled"
DATA_NAMES = "/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/data_cache_3/df_eval.csv"
SMOOTH: bool = True
FILTER: bool = True


def main():
    data_paths_df = pd.read_csv(DATA_NAMES)
    preds_total = pd.DataFrame()
    actual_total = pd.DataFrame()
    majmin_mean = 0
    seventh_mean = 0
    maj_inv_mean = 0
    seventh_inv_mean = 0
    mirex_mean = 0
    csr_mean = 0
    _n = 0
    for i in tqdm(range(len(data_paths_df))):
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

        # Calculate metrics means
        majmin_mean += majmin_accuracy(reference=actual, prediction=preds)
        seventh_mean += seventh_accuracy(reference=actual, prediction=preds)
        maj_inv_mean += maj_inv_accuracy(reference=actual, prediction=preds)
        seventh_inv_mean += seventh_inv_acccuracy(reference=actual, prediction=preds)
        mirex_mean += mirex_accuracy(reference=actual, prediction=preds)
        csr_mean += csr_accuracy(reference=actual, prediction=preds)
        _n += 1

    # Divide means by number of tracks
    majmin_mean = majmin_mean/_n
    seventh_mean = seventh_mean/_n
    maj_inv_mean = maj_inv_mean/_n
    seventh_inv_mean = seventh_inv_mean/_n
    mirex_mean = mirex_mean/_n
    csr_mean = csr_mean/_n

    # Calculate metrics for all tracks
    logger.info("Calculating metrics for all tracks.")
    logger.info("This may take a while..")
    majmin = majmin_accuracy(reference=actual_total, prediction=preds_total)
    seventh = seventh_accuracy(reference=actual_total, prediction=preds_total)
    maj_inv = maj_inv_accuracy(reference=actual_total, prediction=preds_total)
    seventh_inv = seventh_inv_acccuracy(reference=actual_total, prediction=preds_total)
    mirex = mirex_accuracy(reference=actual_total, prediction=preds_total)
    csr = csr_accuracy(reference=actual_total, prediction=preds_total)

    print(100 * '=')
    print("Printing Results")
    print(100 * '=')
    print(f"Major and minor accuracy: {majmin}")
    print(f"Mean Major and minor accuracy: {majmin_mean}")
    print(100*'=')
    print(f"Seventh chords accuracy: {seventh}")
    print(f"Mean Seventh chords accuracy: {seventh_mean}")
    print(100 * '=')
    print(f"Major and minor inversions accuracy: {maj_inv}")
    print(f"Mean Major and minor inversions accuracy: {maj_inv_mean}")
    print(100 * '=')
    print(f"Seventh chords with inversions accuracy: {seventh_inv}")
    print(f"Mean Seventh chords with inversions accuracy: {seventh_inv_mean}")
    print(100 * '=')
    print(f"Mirex accuracy: {mirex}")
    print(f"Mean Mirex accuracy: {mirex_mean}")
    print(100 * '=')
    print(f"Chord Symbol Recall accuracy: {csr}")
    print(f"Mean Chord Symbol Recall accuracy: {csr_mean}")
    print(100 * '=')

    return 0


if __name__ == "__main__":
    gc.collect()
    start = time.time()

    main()

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")