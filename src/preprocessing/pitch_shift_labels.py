import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path
from utils.label_utils import *

from loguru import logger


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


def get_root(chord):
    """Gets root note from mirex (lab) format chord"""
    if "/" in chord and ":" in chord:
        root = chord.split(":", 2)[0]
        rest_of_chord = chord.split(":")[1]
        delim = ":"
    elif "/" in chord:
        root = chord.split("/", 2)[0]
        rest_of_chord = chord.split("/")[1]
        delim = "/"
    elif ":" in chord:
        root = chord.split(":", 2)[0]
        rest_of_chord = chord.split(":")[1]
        delim = ":"
    else:
        root = chord
        rest_of_chord = ''
        delim = ""
    return root, rest_of_chord, delim


class PitchShifterLab:
    def __init__(self, directory, dest_dir, n_steps, pool=None):
        self.direc = directory
        self.dest_dir = dest_dir
        self.n_steps = n_steps
        self.files = None
        self.pooling = False

        if pool:
            self.pooling = True

        self.get_file_list()
        # self.files = ['/home/thanos/Documents/Thesis/all_labels.csv']

    def get_file_list(self):
        """Get list with all wav files' paths"""

        self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.direc) for f in filenames if
                      os.path.splitext(f)[1] == '.lab']

    def load_and_shift(self, file, export_path):
        """load wav, pitch shift and export"""

        df = read_lab(file)
        # df = pd.read_csv(file, delimiter=' ')
        df_shifted = self.pitch_shift_lab(df)
        df_shifted.to_csv(export_path, sep=' ', encoding='utf-8', index=False, header=False)

    def pitch_shift_csv(self, df):
        """Shifts using the converted form of chords (split into features)"""

        df['root'] = df['root'].apply(self.pitch_shift_note)
        df['bass'] = df['bass'].apply(self.pitch_shift_note)
        return df

    def pitch_shift_lab(self, df):
        """Shifts using mirex lab format"""

        df['chord'] = df['chord'].apply(self.pitch_shift_chord)
        return df

    def pitch_shift_chord(self, chord):
        if chord == 'N':
            return 'N'
        root, rest_of_chord, delim = get_root(chord)
        shifted_chord = self.pitch_shift_note(root) + delim + rest_of_chord
        return shifted_chord

    def pitch_shift_note(self, note):
        """
        shifts a note the desired amount of steps
        using 2 steps: C --> D
        """
        if note == 'N':
            return 'N'
        if note in SHARP_TO_FLAT.keys():
            note = SHARP_TO_FLAT[note]
        notes_shifted = shift_list(PITCH_CLASS_NAMES, note)
        return notes_shifted[self.n_steps]

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

        self.load_and_shift(file, self.dest_dir + '/' + album + '/' + song)


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    ARGS = vars(parse_input())
    shifter = PitchShifterLab(**ARGS)
    shifter.run_shifter()

    time_elapsed = time.time() - start
    logger.info(f"Finished, Found {len(shifter.files)} files.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
