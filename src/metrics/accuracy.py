"""Utilities for getting accuracy metrics."""
import pandas as pd
import numpy as np


def get_accuracy(df_pred: pd.DataFrame, df_true: pd.DataFrame) -> pd.DataFrame:
    """Get the accuracy of the predictions.

    :args:
    - df_pred: pd.DataFrame - The DataFrame with the predictions.
    - df_true: pd.DataFrame - The DataFrame with the true values.

    :returns:
    - pd.DataFrame - The DataFrame with the accuracy for each track and each task.
    """
    df_acc = pd.DataFrame()
    for col in df_pred:
        if col not in df_true:
            raise ValueError(f"Column {col} not in true DataFrame.")
        # transform to numpy array
        y_pred = df_pred[col].values
        y_true = df_true[col].values
        # calculate accuracy
        acc = np.mean(y_true == y_pred)
        # add to DataFrame
        df_acc[col] = [acc]
    return df_acc
