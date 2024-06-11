"""
Load encode chunk, reshape, model predict

Full inference pipeline for the 2D CNN model.
This script is used to run the inference pipeline for the given model and data.
"""
import gc
import sys

from src.utils import label_utils

sys.path.append("../src")

from src.utils.train_utils import *


class Inference:
    def __init__(self, model_path, label_col, cache_folder, encoding_dict, save=False):
        """
        Constructor for the Inference class

        :param model: model to be used for inference
        :param data_pats: path to the data to be used for inference
        :param label_col: column (EX. root) to be used for inference
        :param cache_folder: folder to save the output
        :param encoding_dict: Dictionary that includes encodings for the given label
        :param save: boolean to save the output
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model {model_path} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading the model: {e}")
            raise e
        self.label_col = label_col
        self.cache_folder = cache_folder
        self.save = save
        self.encoding_dict = encoding_dict

    def run_inference(self, data_path: str):
        """
        Runs the inference pipeline

        :param data_path: path to the data to be used for inference
        """

        # Load and encode the chunk
        X, y = laod_encode_chunk(data_path, lab_column=self.label_col, encoding_dict=self.encoding_dict, save=self.save)

        # Reshape the features for CNN input
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        # Model predict
        y_pred_id, y = model_predict(self.model, X, y=y, is_2d=True)

        return y_pred_id, y


def main():
    """Runs the inference pipeline for the given model and data"""
    label_col = 'extension_1'
    model_path = f'models/CQT_cnn_{label_col}_30_eval_1_7.h5'
    cache_path = 'predictions'
    encoding_dict = label_utils.EXT_1_ENCODINGS
    data_paths = 'data_cache_4/df_eval.csv'
    save = True

    # Initialize the inference object
    inference = Inference(model_path, label_col, cache_path, encoding_dict, save=False)

    data_paths_df = pd.read_csv(data_paths)

    for i in range(len(data_paths_df)):
        data_path = data_paths_df.iloc[[i]].reset_index()
        song_name, album_name, artist_name = get_data_name(data_path['wav'][0])
        y_pred_id, y = inference.run_inference(data_path)
        if save:
            pd.DataFrame(y_pred_id).to_csv(f"{cache_path}/{label_col}/"
                                           f"y_pred_{artist_name}_{album_name}_{song_name}.csv", index=False)
            pd.DataFrame(y).to_csv(f"{cache_path}/{label_col}/"
                                   f"y_{artist_name}_{album_name}_{song_name}.csv", index=False)


if __name__ == "__main__":
    gc.collect()
    start = time.time()

    main()

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")