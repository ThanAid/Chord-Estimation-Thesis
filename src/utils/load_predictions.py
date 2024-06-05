"""Utility functions to load predictions and actual data from the data path."""
import pandas as pd


def load_track(artist_name: str, album_name: str, song_name: str, data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the track predictions and actual data from the data path.

    Args:
        artist_name (str): The artist name.
        album_name (str): The album name.
        song_name (str): The song name.
        data_path (str): The path to the data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The predictions and actual data DataFrames.
    """
    preds = pd.read_csv(f"{data_path}/y_pred_{artist_name}_{album_name}_{song_name}.csv")
    actual = pd.read_csv(f"{data_path}/y_{artist_name}_{album_name}_{song_name}.csv")

    return preds, actual
