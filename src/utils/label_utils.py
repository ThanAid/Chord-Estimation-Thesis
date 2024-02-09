import pandas as pd
import matplotlib.pyplot as plt

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

SEMITONE_INTERVALS = {
    '#1': 1,
    'bb2': 0, 'b2': 1, '2': 2, '#2': 3,
    'bb3': 2, 'b3': 3, '3': 4, '#3': 5,
    'b4': 4, '4': 5, '#4': 6,
    'b5': 6, '5': 7, '#5': 8,
    'bb6': 7, 'b6': 8, '6': 9, '#6': 10,
    'bb7': 9, 'b7': 10, '7': 11, '#7': 12,
    'b8': 11, '8': 12
}

EQUIVALENTS = {
    'b9': 'b2', '9': '2', 'bb9': 'bb2', '#9': '#2',
    'b10': 'b3', '10': '3', 'bb10': 'bb3', '#10': '#3',
    'b11': 'b4', '11': '4', '#11': '#4',
    'b13': 'b5', '13': '5', '#13': '#5',
    'b14': 'b6', '14': '6', 'bb14': 'bb6', '#14': '#6',
    'b15': 'b7', '15': '7', 'bb15': 'bb7', '#15': '#7',
    'b16': 'b8', '16': '8', 'bb16': 'bb8'
}

NOTE_ENCODINGS = {'N': 0, 'C': 1, 'Db': 2, 'D': 3, 'Eb': 4, 'E': 5, 'F': 6,
                  'Gb': 7, 'G': 8, 'Ab': 9, 'A': 10, 'Bb': 11, 'B': 12}

TRIAD_ENCODINGS = {'N': 0, 'maj': 1, 'min': 2, 'dim': 3, 'aug': 4, 'sus2': 5, 'sus4': 6}


def shift_list(lst, item):
    """Shifts a list in the index of the item selected"""
    idx = lst.index(item)
    return lst[idx:] + lst[:idx]


def read_lab(lbl_path):
    """read a .lab mirex formatted file and return a df"""

    return pd.read_csv(lbl_path, delimiter=' ', names=['start', 'end', 'chord'], index_col=False)


def read_converted_chords(converted_path):
    """Reads a converted chord file and returns a pandas dataframe"""
    columns = ['root', 'bass', 'triad', 'extension_1', 'extension_2']

    return pd.read_csv(converted_path, delimiter=' ', names=columns, index_col=False)


def sharp_to_flat_semi(sharp):
    """takes for example #5 and converts it to b6"""
    sharp_int = int(''.join(i for i in sharp if i.isdigit()))
    return 'b' + str(sharp_int + 1)


def create_plots_for_columns(data):
    # Check if the data contains columns other than the index
    if len(data.columns) <= 1:
        print("The data does not contain columns to create plots.")
        return

    # Create subplots to display multiple plots together
    num_columns = len(data.columns)
    num_rows = 2  # Number of rows for subplots (adjust as needed)
    num_cols = (num_columns + num_rows - 1) // num_rows  # Number of columns for subplots

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 8))

    # Flatten the axes to iterate over them
    axes = axes.flatten()

    # Loop through the columns and create plots for each
    for i, column in enumerate(data.columns):
        if column != 'index':
            ax = axes[i]
            x_data = data[column].value_counts()
            if len(x_data) > 30:
                x_data = x_data[:30]
            x_data.plot(kind='bar', ax=ax)
            ax.set_title(f'Distribution of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')

    # Remove any unused subplots
    for i in range(num_columns, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
