import sys

sys.path.append("../src")

# import local libs
from src.utils.transfer_learning import *


def main():
    logger.info("Starting up..")

    model_path = 'models/CQT_cnn_root_10.h5'
    label_col = 'extension_2'
    cache_path = 'data_cache'
    dest_model = f"models/CQT_cnn_{label_col}_10.h5"

    TransferLearning(model_path, label_col, cache_path, dest_model=dest_model).run_pipeline()


if __name__ == "__main__":
    gc.collect()
    start = time.time()

    main()

    time_elapsed = time.time() - start

    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
