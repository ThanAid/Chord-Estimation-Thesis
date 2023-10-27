import pandas as pd
import numpy as np
from label_utils import *

PITCH_CLASS_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
FLAT_TO_SHARP = {
    'Db': 'C#',
    'Eb': 'D#',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#'
}

SHARP_TO_FLAT = {
    'C#': 'Db',
    'D#': 'Eb',
    'F#': 'Gb',
    'G#': 'Ab',
    'A#': 'Bb'
}

SEMITONE_STEPS = {
    'b2': 1,
    '2': 2,
    'b3': 3,
    '3': 4,
    'b4': 4,
    '4': 5,
    'b5': 6,
    '5': 7,
    'b6': 8,
    '6': 9,
    'b7': 9,
    '7': 10,
    'b8': 11,
    '8': 12
}


class AdaptLabels:
    def __init__(self, label_path, n_steps):
        self.time_labels = read_lab(label_path)
        self.n_steps = n_steps
        self.duration = self.get_duration()
        self.timestep = self.get_timestep()
        self.timestamps = np.linspace(0, self.duration, num=n_steps)
        self.labels = self.adapt_labels()

    def get_duration(self):
        """returns duration of track"""

        return self.time_labels['end'].iloc[-1]

    def get_timestep(self):
        """returns the duration of each time step"""

        return self.duration / self.n_steps

    def adapt_labels(self):
        """transforms time based labels to frame based labels"""

        i = 0
        labels = []
        for timestamp in self.timestamps:
            if timestamp <= self.time_labels["end"][i]:
                labels.append((timestamp, self.time_labels["chord"][i]))
            else:
                i += 1
                labels.append((timestamp, self.time_labels["chord"][i]))

        return labels


# class Chord:
#     """Chord object to store properties"""
#     def __init__(self):
#         self.root = None
#         self.bass = None
#         self.triad = None


class CovertLab:

    @staticmethod
    def convert_chordlab_df(df, label_col='label'):
        """
        Converts a DataFrame with chord segment annotations - add columns with
        binary pitch classes and with root and bass pitch classes.

        Input: a DataFrame with columns including `label_col`, typically
               ('start', 'end', 'label')
        Output: a DataFrame with added columns
               ('root', 'bass', 'C', 'Db', ..., 'Bb', 'B')
        """
        df['root'] = df[label_col].apply(lambda i: CovertLab.get_root(i))
        df['bass'] = df[label_col].apply(lambda i: CovertLab.get_bass(i))
        df['triad'] = df[label_col].apply(lambda i: CovertLab.get_triad(i))
        df['extension_1'] = df[label_col].apply(lambda i: CovertLab.get_extension_1(i))
        df['extension_2'] = df[label_col].apply(lambda i: CovertLab.get_extension_2(i))
        # TODO: Better just use columns 'pc_0', 'pc_1', ..., 'pc_12' that can be
        # later selected more easily.
        # for pc, name in enumerate(PITCH_CLASS_NAMES):
        #     df[name] = parsed_labels.apply(lambda l: l.tones_binary[pc])
        return df

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
        return root

    @staticmethod
    def get_bass(chord):
        root = CovertLab.get_root(chord)  # TODO: optimize that
        if "/" in chord:
            bass = chord.split("/")[1]
            bass = CovertLab.translate_bass(root, bass)
            # TODO: translate bass to actual note
        else:
            bass = root
        return bass

    @staticmethod
    def get_triad(chord):
        if "maj" in chord or "min" not in chord:
            triad = 'Major'
        elif "min" in chord:
            triad = "Minor"
        else:
            triad = "N"
        # TODO: include Diminished , Augmented , Sus 2, Sus4
        return triad

    @staticmethod
    def get_extension_1(chord):
        # If the chord is an inversion keep the base part of the chord
        if "/" in chord:
            chord = chord.split("/")[0]
        if 'maj6' in chord:
            return 'maj6'
        elif 'maj7' in chord:
            return 'maj7'
        elif '7' in chord or '9' in chord:
            return 'min7'
        elif 'dim7' in chord:
            return 'dim7'
        else:
            return 'N'

    @staticmethod
    def get_extension_2(chord):
        # If the chord is an inversion keep the base part of the chord
        if "/" in chord:
            chord = chord.split("/")[0]
        if '9' in chord:
            return 'maj9'
        else:
            return 'N'

    @staticmethod
    def translate_bass(root, bass):
        """Translates bass to actual note based on root"""

        # Get rid of #
        if root in SHARP_TO_FLAT.keys():
            root = SHARP_TO_FLAT[root]

        shifted_pitches = shift_list(PITCH_CLASS_NAMES, root)
        # if not bass.isdigit():
        #     # if b7 convert to a semitone back
        #     bass_int = int(''.join(i for i in bass if i.isdigit()))
        if bass in SEMITONE_STEPS.keys():
            return shifted_pitches[SEMITONE_STEPS[bass]]
        #     return shifted_pitches[(int(bass_int) * 2 - 3) % 12]  # (bass-1)*2 - 1(b) because of idx starting from 0
        # return shifted_pitches[(int(bass) * 2 - 2) % 12]  # (bass-1)*2  because of idx starting from 0
        else:
            return 'skata'


if __name__ == "__main__":
    label_path = "/home/thanos/Documents/Thesis/all_labels.csv"
    labels = pd.read_csv(label_path, on_bad_lines='skip', index_col=False)
    labels = labels.drop_duplicates('chord')
    # labels = read_lab(label_path)

    converted_test = CovertLab.convert_chordlab_df(labels, label_col='chord')
    print()
