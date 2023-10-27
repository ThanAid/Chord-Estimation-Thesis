import pandas as pd

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

EQUIVALENTS = {
    'b9': 'b2',
    '9': '2',
    'b10': 'b3',
    '10': '3',
    'b11': 'b4',
    '11': '4',
    'b13': 'b5',
    '13': '5',
    'b14': 'b6',
    '14': '6',
    'b15': 'b7',
    '15': '7',
    'b16': 'b8',
    '16': '8'
}


def shift_list(lst, item):
    """Shifts a list in the index of the item selected"""
    idx = lst.index(item)
    return lst[idx:] + lst[:idx]


def read_lab(lbl_path):
    """read a .lab mirex formatted file and return a df"""

    return pd.read_csv(lbl_path, delimiter=' ', names=['start', 'end', 'chord'], index_col=False)


def sharp_to_flat_semi(sharp):
    """takes for example #5 and converts it to b6"""
    sharp_int = int(''.join(i for i in sharp if i.isdigit()))
    return 'b' + str(sharp_int + 1)
