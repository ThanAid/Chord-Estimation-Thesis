import gc

from src.post_processing.assemble_chord import AssembleChord
from src.utils.train_utils import *


def assemble_preds():
    """Assemble the predictions into one csv file based on all the tracks on eval set."""
    prediction_folder = "/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/predictions"
    data_paths = 'data_cache_3/df_eval.csv'
    data_paths_df = pd.read_csv(data_paths)

    for i in range(len(data_paths_df)):
        # Load and encode the chunk
        data_path = data_paths_df.iloc[[i]].reset_index()
        song_name, album_name, artist_name = get_data_name(data_path['wav'][0])

        # Assemble the predictions into one csv file
        file_name_pred = f"y_pred_{artist_name}_{album_name}_{song_name}.csv"
        save_path_pred = prediction_folder + "/assembled/" + file_name_pred

        file_name_y = f"y_{artist_name}_{album_name}_{song_name}.csv"
        save_path_y = prediction_folder + "/assembled/" + file_name_y

        assembler = AssembleChord(prediction_folder, file_name_pred, save_path_pred)
        assembler.assemble()
        assembler = AssembleChord(prediction_folder, file_name_y, save_path_y)
        assembler.assemble()


if __name__ == "__main__":
    gc.collect()
    start = time.time()

    assemble_preds()

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
