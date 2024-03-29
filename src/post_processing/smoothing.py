"""
Post-processing functions for the chords.

This module contains functions that are used to post process the chords.
"""
import pandas as pd
from collections import Counter


# TODO: REALLY IMPORTANT! FIRST SMOOTH THEN FILTER
def get_closest_triad(frames: pd.DataFrame, i: int, direction: str = "left"):
    """
    Get the closest triad to the given index.

    Args:
        frames (pd.DataFrame): DataFrame containing the chords.
        i (int): Index of the chord to be used.
        direction (str): Direction to search for the closest triad. Can be "left" or "right".

    Returns:
        int: Closest triad to the given index.
    """
    if i < 0:
        return 1, i  # assign major if nothing is found
    if i >= len(frames):
        return 1, i  # assign major if nothing is found
    if frames.iloc[i, 2] != 0:
        return frames.iloc[i, 2], i
    else:
        if direction == "left":
            triad, position = get_closest_triad(frames, i - 1, direction=direction)
        else:
            triad, position = get_closest_triad(frames, i + 1, direction=direction)
    return triad, position


def chord_filter(frames: pd.DataFrame) -> pd.DataFrame:
    """
    Filter each row (frame) based on some rules.

     If root is None, then the whole chord is None.
     If bass is None, then bass is root.
     if triad is None then triad is gets closest value.

    Args:
        frames (pd.DataFrame): DataFrame containing the chords to be filtered.

    Returns:
        pd.DataFrame: DataFrame containing the filtered chords.
     """
    output = frames.copy()  # don't overwrite the input
    for i in range(len(frames)):
        if frames.iloc[i, 0] == 0:
            output.iloc[i] = 0
        elif frames.iloc[i, 1] == 0:
            output.iloc[i, 1] = frames.iloc[i, 0]
            if frames.iloc[i, 2] == 0:
                tr_l, left = get_closest_triad(frames, i, direction="left")
                tr_r, rigth = get_closest_triad(frames, i, direction="right")
                if abs(i - left) < abs(i - rigth):
                    output.iloc[i, 2] = tr_l
                else:
                    output.iloc[i, 2] = tr_r
        elif frames.iloc[i, 2] == 0:
            tr_l, left = get_closest_triad(frames, i, direction="left")
            tr_r, rigth = get_closest_triad(frames, i, direction="right")
            if abs(i - left) < abs(i - rigth):
                output.iloc[i, 2] = tr_l
            else:
                output.iloc[i, 2] = tr_r

    return output


def smooth_column(df: pd.DataFrame, column: str, window_size: int, in_place: bool = False) -> pd.DataFrame:
    """
    Smooth the specified column in the DataFrame by replacing values with the majority value within a given window size.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to be smoothed.
        window_size (int): Size of the window to consider for smoothing.
        in_place (bool): If True, the DataFrame is modified in place. Otherwise, a new DataFrame is returned.
    """
    if not in_place:
        output = df.copy()  # don't overwrite the input
    else:
        output = df
    smoothed_values = []

    # TODO: think better about the implementation
    for i in range(len(df)):
        start_index = max(0, i - window_size)
        end_index = min(len(df), i + window_size + 1)
        window_values = df[column].iloc[start_index:end_index]

        value_counts = Counter(window_values)
        most_common_value, most_common_count = value_counts.most_common(1)[0]

        if most_common_count >= window_size:
            smoothed_values.append(most_common_value)
        else:
            smoothed_values.append(df[column].iloc[i])

    output[column] = smoothed_values
    return output
