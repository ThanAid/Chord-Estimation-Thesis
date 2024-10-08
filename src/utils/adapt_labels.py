import time

import numpy as np
from loguru import logger
import sys

sys.path.append("../src")

from src.utils.label_utils import *


class AdaptLabels:
    def __init__(self, label_path, n_steps, duration=None):
        self.time_labels = read_lab(label_path)
        self.n_steps = n_steps

        # If duration of the track is given we use that, else we calculate it
        if duration:
            self.duration = duration
        else:
            self.duration = self.get_duration()
        self.timestep = self.get_timestep()
        self.timestamps = np.linspace(0, self.duration, num=n_steps)
        self.labels = self.adapt_labels()

    def get_duration(self):
        """returns duration of track"""

        return float(self.time_labels['end'].iloc[-1])

    def get_timestep(self):
        """returns the duration of each time step"""

        return self.duration / self.n_steps

    def adapt_labels(self):
        """transforms time based labels to frame based labels"""

        i = 0
        labels = []
        for timestamp in self.timestamps:
            try:
                if timestamp <= self.time_labels["end"][i]:
                    labels.append((timestamp, self.time_labels["chord"][i]))
                else:
                    i += 1
                    labels.append((timestamp, self.time_labels["chord"][i]))

            # catch key error if there is extra frames in the audiofile (1-2)
            # and fill them with N - sync audiofile to labels
            except KeyError:
                labels.append((timestamp, 'N'))
        return labels


# class Chord:
#     """Chord object to store properties"""
#     def __init__(self):
#         self.root = None
#         self.bass = None
#         self.triad = None


class ConvertLab:
    def __init__(self, label_path, label_col='label', dest=None, is_df=False):
        """
        Extracts label features like root, bass etc.

        :param label_path: the paths of the labels
        :param label_col: column name that contains the labels
        :param dest: if given then it exports to csv
        :param is_df: if False then label_path is a path, if True then label_path is the actual dataframe
        """
        if not is_df:
            self.df = pd.read_csv(label_path, on_bad_lines='skip', index_col=False)
        else:
            self.df = label_path

        self.label_col = label_col
        self.dest = dest

        self.convert_chordlab_df()

        if self.dest:
            self.export_to_csv()

    def convert_chordlab_df(self):
        """
        Converts a DataFrame with chord segment annotations - add columns with
        binary pitch classes and with root and bass pitch classes.

        Input: a DataFrame with columns including `label_col`, typically
               ('start', 'end', 'label')
        Output: a DataFrame with added columns
               ('root', 'bass', 'C', 'Db', ..., 'Bb', 'B')
        """
        self.df['root'] = self.df[self.label_col].apply(lambda i: ConvertLab.get_root(i))
        self.df['bass'] = self.df[self.label_col].apply(lambda i: ConvertLab.get_bass(i))
        self.df['triad'] = self.df[self.label_col].apply(lambda i: ConvertLab.get_triad(i))
        self.df['extension_1'] = self.df[self.label_col].apply(lambda i: ConvertLab.get_extension_1(i))
        self.df['extension_2'] = self.df[self.label_col].apply(lambda i: ConvertLab.get_extension_2(i))

    def export_to_csv(self):
        self.df.to_csv(self.dest, sep=' ', encoding='utf-8', index=False)

    @staticmethod
    def get_root(chord):
        """Gets root note from mirex (lab) format chord"""
        if "/" in chord and ":" in chord:
            root = chord.split(":")[0]
        elif "/" in chord:
            root = chord.split("/")[0]
        elif ":" in chord:
            root = chord.split(":")[0]
        else:
            root = chord

        # Get rid of sharps
        if root in SHARP_TO_FLAT.keys():
            root = SHARP_TO_FLAT[root]

        return root

    @staticmethod
    def get_bass(chord):
        root = ConvertLab.get_root(chord)
        if "/" in chord:
            bass = chord.split("/")[1]
            bass = ConvertLab.translate_bass(root, bass)
        else:
            bass = root
        return bass

    @staticmethod
    def get_triad(chord):
        if "dim" in chord:
            triad = "dim"
        elif "aug" in chord:
            triad = "aug"
        elif "sus4" in chord:
            triad = "sus4"
        elif "sus2" in chord:
            triad = "sus2"
        elif 'N' in chord:
            triad = "N"
        elif "maj" in chord or "min" not in chord:
            triad = 'maj'
        elif "min" in chord:
            triad = "min"
        else:
            triad = "N"

        return triad

    @staticmethod
    def get_extension_1(chord):
        # If the chord is an inversion keep the base part of the chord
        if "/" in chord:
            chord = chord.split("/")[0]
        if 'maj6' in chord:
            return 'maj6'
        elif 'min6' in chord:
            return 'maj6'
        elif 'maj7' in chord:
            return 'maj7'
        elif 'hdim7' in chord:
            return 'hdim7'
        elif 'dim7' in chord:
            return 'dim7'
        elif '7' in chord:
            return '7'
        elif '9' in chord:
            return 'min7'
        else:
            return 'N'

    @staticmethod
    def get_extension_2(chord):
        # If the chord is an inversion keep the base part of the chord
        if "/" in chord:
            chord = chord.split("/")[0]
        if '9' in chord:
            return '9'
        else:
            return 'N'

    @staticmethod
    def translate_bass(root, bass):
        """Translates bass to actual note based on root"""

        # Get rid of sharps
        if root in SHARP_TO_FLAT.keys():
            root = SHARP_TO_FLAT[root]

        shifted_pitches = shift_list(PITCH_CLASS_NAMES, root)

        if bass not in SEMITONE_INTERVALS.keys():
            bass = EQUIVALENTS[bass]
        return shifted_pitches[SEMITONE_INTERVALS[bass]]


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    label_path = "/home/thanos/Documents/Thesis/all_labels.csv"
    dest = "/home/thanos/Documents/Thesis/all_labels_converted_7.csv"
    labels = pd.read_csv(label_path, on_bad_lines='skip', index_col=False)
    labels = labels.drop_duplicates('chord')
    labels = read_lab(label_path)

    adapted_labs = ConvertLab(label_path, label_col='chord', dest=dest, is_df=False)
    time_elapsed = time.time() - start
    # logger.info(f"Finished, Found {len(converted_test.df)} chords.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")

