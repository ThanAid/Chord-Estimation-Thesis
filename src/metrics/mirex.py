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

# ================================= MIREX METRICS =================================


def majmin_row(row):
    """Return true if the row is a maj or min chord."""
    if (row[2] == 0 or row[2] == 1 or row[2] == 2) and (row[0] == row[1]):
        return True
    return False


def seventh_row(row):
    """Return true if the row is a seventh chord.

    Seventh chords: {N, maj, min, maj7, min7, 7};
    """
    if ((row[2] == 0 or row[2] == 1 or row[2] == 2) and (row[3] == 0
            or row[3] == 4 or row[3] == 5) and (row[0] == row[1])):
        return True
    return False


def maj_inv_row(row):
    """Return true if the row is a maj or inv chord."""
    if row[2] == 0 or row[2] == 1 or row[2] == 2:
        return True
    return False


def seventh_inv_row(row):
    """Return true if the row is a seventh or inv chord."""
    if ((row[2] == 0 or row[2] == 1 or row[2] == 2) and (row[3] == 0
            or row[3] == 4 or row[3] == 5)):
        return True
    return False


def majmin_accuracy(reference: pd.DataFrame, prediction: pd.DataFrame):
    """Mirex metric that includes only maj and min chords.

    Major and minor: {N, maj, min}
    """
    correct = 0
    samples = 0
    for i in range(len(reference)):
        if majmin_row(reference.iloc[i]) and majmin_row(prediction.iloc[i]):
            if reference.iloc[i][2] == prediction.iloc[i][2] and reference.iloc[i][0] == prediction.iloc[i][0]:
                correct += 1
            samples += 1
    return correct / samples


def seventh_accuracy(reference: pd.DataFrame, prediction: pd.DataFrame):
    """Mirex metric that includes only seventh chords.

    Seventh chords: {N, maj, min, maj7, min7, 7};
    """
    correct = 0
    samples = 0
    for i in range(len(reference)):
        if seventh_row(reference.iloc[i]) and seventh_row(prediction.iloc[i]):
            if (reference.iloc[i][2] == prediction.iloc[i][2]
                    and reference.iloc[i][3] == prediction.iloc[i][3]
                    and reference.iloc[i][0] == prediction.iloc[i][0]):
                correct += 1
            samples += 1
    return correct / samples


def maj_inv_accuracy(reference: pd.DataFrame, prediction: pd.DataFrame):
    """Mirex metric that includes only maj and inv chords.

    Major and minor with inversions: {N, maj, min, maj/3, min/b3, maj/5, min/5}
    """
    correct = 0
    samples = 0
    for i in range(len(reference)):
        if maj_inv_row(reference.iloc[i]) and maj_inv_row(prediction.iloc[i]):
            if (reference.iloc[i][2] == prediction.iloc[i][2]
                    and reference.iloc[i][1] == prediction.iloc[i][1]
                    and reference.iloc[i][0] == prediction.iloc[i][0]):
                correct += 1
            samples += 1
    return correct / samples


def seventh_inv_acccuracy(reference: pd.DataFrame, prediction: pd.DataFrame):
    """Mirex metric that includes only seventh and inv chords.

    Seventh chords with inversions: {N, maj, min, maj7, min7, 7,
    maj/3, min/b3, maj7/3, min7/b3, 7/3, maj/5, min/5, maj7/5, min7/5,
    7/5, maj7/7, min7/b7, 7/b7}
    """
    correct = 0
    samples = 0
    for i in range(len(reference)):
        if seventh_inv_row(reference.iloc[i]) and seventh_inv_row(prediction.iloc[i]):
            if (reference.iloc[i][2] == prediction.iloc[i][2]
                    and reference.iloc[i][3] == prediction.iloc[i][3]
                    and reference.iloc[i][1] == prediction.iloc[i][1]
                    and reference.iloc[i][0] == prediction.iloc[i][0]):
                correct += 1
            samples += 1
    return correct / samples


def mirex_accuracy(reference: pd.DataFrame, prediction: pd.DataFrame):
    """Mirex metric that includes all chords."""
    correct = 0
    samples = 0
    for i in range(len(reference)):
        if seventh_inv_row(reference.iloc[i]):
            if prediction.iloc[i][3] == 0:  # don't count none 7th as correct note
                row_correct = sum([reference.iloc[i][j] == prediction.iloc[i][j] for j in range(3)])
            else:
                row_correct = sum([reference.iloc[i][j] == prediction.iloc[i][j] for j in range(4)])
            if row_correct >= 3:
                correct += 1
            samples += 1

    return correct / samples


def csr_accuracy(reference: pd.DataFrame, prediction: pd.DataFrame):
    """Chord Symbol Recall (CSR) metric."""
    correct = 0
    samples = 0
    for i in range(len(reference)):
        if seventh_inv_row(reference.iloc[i]):
            row_correct = sum([reference.iloc[i][j] == prediction.iloc[i][j] for j in range(4)])
            if row_correct == 4:
                correct += 1
            samples += 1

    return correct / samples