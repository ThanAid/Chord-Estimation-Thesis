import argparse
import os
import time

from loguru import logger

from label_utils import *


def parse_input(args=None):
    """
       Parse cmd line arguments
       :param args: The command line arguments provided by the user
       :return: The parsed input Namespace
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--directory", type=str, action="store", metavar="directory",
                        required=True)
    parser.add_argument("-dest", "--destination", type=str, action="store", metavar="destination",
                        required=False)

    return parser.parse_args(args)


class GetLabels:
    def __init__(self, directory, destination):
        self.direc = directory
        self.dest = destination
        self.files = None
        self.labels = pd.DataFrame(columns=['chord'])

        self.get_file_list()
        self.get_labels()

        # Export if destination is given
        if self.dest:
            self.export_to_csv()

    def get_file_list(self):
        self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.direc) for f in filenames if
                      os.path.splitext(f)[1] == '.lab']

    def get_labels(self):
        for file in self.files:
            df = read_lab(file).drop(columns=['start', 'end'])
            self.labels = pd.concat([self.labels, df], axis=0, ignore_index=True)

    def export_to_csv(self):
        self.labels.to_csv(self.dest, sep=' ', encoding='utf-8', index=False)


if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    ARGS = vars(parse_input())
    labels = GetLabels(**ARGS)

    time_elapsed = time.time() - start
    logger.info(f"Finished, Found {len(labels.files)} files.")
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
