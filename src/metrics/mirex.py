"""Metric for MIREX evaluation."""
import pandas as pd


def chord_parts_precision(reference: pd.DataFrame, prediction: pd.DataFrame, n_parts: int = 3):
    """Based on the number of parts given return the precision of the prediction on those parts.

    For example if n_parts=3 the precision will be calculated on root, bass and triad and for
    the row to be considered correctly predicted all 3 must be correctly predicted.

    Args:
        reference (pd.DataFrame): Reference pd.DataFrames of chords split into root, bass, triad, ext1 and exr2.
        prediction (pd.DataFrame): Prediction pd.DataFrames of chords split into root, bass, triad, ext1 and exr2.
        n_parts (int): Number of parts to consider for precision calculation.

    Returns:
        float: Precision of the prediction on the given number of parts.
    """
    # compare each row of the two dataframes
    correct = 0
    for i in range(len(reference)):
        if reference.iloc[i, :n_parts].equals(prediction.iloc[i, :n_parts]):
            correct += 1

    return correct / len(reference)


def mirex_precision(reference, prediction):
    """Compute MIREX precision.

    Args:
        reference (pd.DataFrame): Reference pd.DataFrames of chords split into root, bass, triad, ext1 and exr2.
        prediction (pd.DataFrame): Prediction pd.DataFrames of chords split into root, bass, triad, ext1 and exr2.

    Returns:
        float: MIREX precision.

    """
    # TODO: mirex only has maj-min on triad so i should filter all other b4 comparing.
    pass
