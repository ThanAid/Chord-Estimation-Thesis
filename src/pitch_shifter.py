import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

from loguru import logger

from utils.audio_utils import *


def parse_input(args=None):
    """
       Parse cmd line arguments
       :param args: The command line arguments provided by the user
       :return: The parsed input Namespace
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--directory", type=str, action="store", metavar="directory",
                        required=True)
    parser.add_argument("-dest", "--dest_dir", type=str, action="store", metavar="dest_dir",
                        required=True)
    parser.add_argument("-n", "--n_steps", type=int, action="store", metavar="n_steps",
                        required=True)
    parser.add_argument("-p", "--pool", action="store_true",
                        required=False)

    return parser.parse_args(args)


class PitchShifter:
    def __init__(self, directory, dest_dir, n_steps, pool=None):
        self.direc = directory
        self.dest_dir = dest_dir
        self.n_steps = n_steps
        self.files = None
        self.pooling = False

        if pool:
            self.pooling = True

        self.get_file_list()

    def get_file_list(self):
        """Get list with all wav files' paths"""

        self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.direc) for f in filenames if
                      os.path.splitext(f)[1] == '.wav']

    @staticmethod
    def load_and_shift(file, export_path, n_steps):
        """load wav, pitch shift and export"""

        y, sr = load_wav(file, sampling_rate=44100)
        y_shifted = pitch_shift(y, sr=sr, n_steps=n_steps)
        export_wav(export_path, y_shifted, sampling_rate=sr)

    def run_shifter(self):
        """runs the pitch shifter"""
        if self.pooling:
            logger.info('Pooling started')
            with Pool(12) as p:
                p.map(self.shifter_iter, self.files)
        else:
            for file in self.files:
                self.shifter_iter(file)

    def shifter_iter(self, file):
        """iterable method for shifting"""
        logger.info(f"Shifting {file}...")
        album = file.split('/')[-2]
        song = file.split('/')[-1]
        # If folder is non-existent, create it
        Path(self.dest_dir + '/' + album).mkdir(parents=True, exist_ok=True)

        self.load_and_shift(file, self.dest_dir + '/' + album + '/' + song, n_steps=self.n_steps)


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    ARGS = vars(parse_input())
    shifter = PitchShifter(**ARGS)
    shifter.run_shifter()

    time_elapsed = time.time() - start
    logger.info(f"Finished, Found {len(shifter.files)} files.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
