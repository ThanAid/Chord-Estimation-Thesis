import sys

sys.path.append("../src")

# import local libs
from src.utils.transfer_learning import *
from src.utils import label_utils


def main():
    logger.info("Starting up..")

    model_path = 'models/CQT_cnn_root_40_w.h5'
    label_col = 'triad'
    cache_path = 'data_cache'
    dest_model = f"models/CQT_cnn_{label_col}_20_5.h5"
    encoding_dict = label_utils.TRIAD_ENCODINGS

    # Initialize transfer learner object
    transfer_learner = TransferLearning2D(model_path, label_col, cache_path, dest_model=dest_model, epochs=30,
                                          batch_size=16, label_mapping=encoding_dict, use_weights=False,
                                          n_sliced_layers=10)
    transfer_learner.run_pipeline()


if __name__ == "__main__":
    gc.collect()
    start = time.time()

    main()

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
