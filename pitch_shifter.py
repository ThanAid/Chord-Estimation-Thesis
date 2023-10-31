import argparse
import os
import time

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

    return parser.parse_args(args)


class PitchShifter:
    def __init__(self, directory, dest_dir, n_steps):
        self.direc = directory
        self.dest_dir = dest_dir
        self.n_steps = n_steps
        self.files = None

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

        for file in self.files:
            logger.info(f"Shifting {file}...")
            #TODO: fix this mess album+song and mkdir album
            album_song = '/'.join(file.split('/')[-2:])
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.load_and_shift(file, self.dest_dir + '/' + album_song, n_steps=self.n_steps)


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    ARGS = vars(parse_input())
    shifter = PitchShifter(**ARGS)
    shifter.run_shifter()

    time_elapsed = time.time() - start
    logger.info(f"Finished, Found {len(shifter.files)} files.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
