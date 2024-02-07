import sys

sys.path.append("../src")

# import local libs
from src.utils.transfer_learning import *


def main():
    logger.info("Starting up..")

    model_path = 'models/CQT_cnn_root_40_w.h5'
    label_col = 'bass'
    cache_path = 'data_cache'
    dest_model = f"models/CQT_cnn_{label_col}_20_2.h5"

    # Initialize transfer learner object
    transfer_learner = TransferLearning2D(model_path, label_col, cache_path, dest_model=dest_model, epochs=20,
                                          batch_size=16, label_mapping=label_utils.NOTE_ENCODINGS, use_weights=True)
    transfer_learner.run_pipeline()


if __name__ == "__main__":
    gc.collect()
    start = time.time()

    main()

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
