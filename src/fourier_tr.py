import argparse
import os

import pandas as pd
from multiprocessing import Pool
import time
from utils import audio_utils
import adapt_labels
from loguru import logger
import numpy as np


def parse_input(args=None):
    """
       Parse cmd line arguments
       :param args: The command line arguments provided by the user
       :return: The parsed input Namespace
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-paths", "--paths_txt", type=str, action="store", metavar="paths_txt",
                        required=True)
    parser.add_argument("-d", "--dest_txt", type=str, action="store", metavar="dest_txt",
                        required=True, help="destination path of new txt containing paths for the transformed dataset")
    parser.add_argument("-p", "--pool", action="store_true",
                        required=False)
    return parser.parse_args(args)


class FourierTransf:
    def __init__(self, paths_txt, dest_txt, pool=False):
        self.paths_df = pd.read_csv(paths_txt, delimiter=' ', index_col=False, names=['wav', 'labels'], header=None)
        self.dest_txt = dest_txt
        self.pooling = False
        self.trsn_paths_df = pd.DataFrame(columns=['audio_csv', 'labels'])

        if pool:
            self.pooling = True

        self.run_fourier_trn()

    @staticmethod
    def tansf_labels(label_path, stft_length):
        """Transforms labels from time format to frame format"""

        adapted_labs = adapt_labels.AdaptLabels(label_path, stft_length)
        return adapted_labs

    @staticmethod
    def transf_audio(audio_path, sr=44100, hop_size=4410, win_size=8192, q=True):
        """
        Transforms audio using stft

        :param audio_path:
        :param sr:
        :param hop_size:
        :param win_size:
        :param q: if True performs Constant Q Transform to calculate Pitch Class Profile (PCP), normalized
        :return:
        """

        y, sr = audio_utils.load_wav(audio_path, sampling_rate=sr)

        stft = audio_utils.stf_transform(y, hop_size=hop_size, win_size=win_size, q=q)
        return stft, sr

    def transform_and_export(self, track_path):
        """
        load wav, pitch shift and export

        :param track_path: zipped audio path and label path
        Ex. track_path = zip(self.paths_df['wav'], self.paths_df['labels'])
        """

        audio_path, label_path = track_path

        logger.info(f"Transforming:\n{track_path}")

        trns_audio_path = audio_path.split('.wav')[0] + '.csv'
        trns_label_path = label_path[:-4] + "_TRNS.csv"

        if os.path.exists(trns_audio_path):
            logger.info(f"Skipping {trns_audio_path} because it already exists...")
        else:
            ft, sr = self.transf_audio(audio_path=audio_path, q=True)
            logger.info(f"Exporting {trns_audio_path}.\n")

            np.savetxt(trns_audio_path, ft, delimiter=",")

        if os.path.exists(trns_label_path):
            logger.info(f"Skipping {trns_label_path} because it already exists...")
        else:
            adapted_labels = self.tansf_labels(label_path=label_path, stft_length=ft.shape[0])
            # TODO: shape[0] wont work for stft i think (Needs check)
            pd.DataFrame(adapted_labels.labels).to_csv(trns_label_path, sep=' ', encoding='utf-8', index=False,
                                                       header=False)

        # New row data to append
        new_row = {'audio_csv': trns_audio_path, 'labels': trns_label_path}

        # Append the new row to the DataFrame
        self.trsn_paths_df = pd.concat([self.trsn_paths_df, pd.DataFrame([new_row])], ignore_index=True)

    def run_fourier_trn(self):
        """Runs the fourier transform for all pairs of wav-label"""

        if self.pooling:
            logger.info('Pooling started')
            with Pool(12) as p:
                p.map(self.transform_and_export, zip(self.paths_df['wav'], self.paths_df['labels']))
        else:
            for track_path in zip(self.paths_df['wav'], self.paths_df['labels']):
                self.transform_and_export(track_path)

        # Store the paths of transformed dataset to txt.
        logger.info("Saving paths to txt...")
        self.trsn_paths_df.to_csv(self.dest_txt, header=None, index=None, sep=' ')


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    ARGS = vars(parse_input())
    FourierTransf(**ARGS)

    time_elapsed = time.time() - start
    # logger.info(f"Finished, Found {len(converted_test.df)} chords.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")