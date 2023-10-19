# import required modules
import subprocess
import argparse
from pydub import AudioSegment
import os
from loguru import logger


def parse_input(args=None):
    """
       Parse cmd line arguments
       :param args: The command line arguments provided by the user
       :return: The parsed input Namespace
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--directory", type=str, action="store", metavar="directory",
                        required=True)
    parser.add_argument("-mn", "--mono", type=bool, action="store", metavar="mono",
                        required=False)

    return parser.parse_args(args)


class ConvertAudio:
    def __init__(self, direc, mono=None):
        self.direc = direc
        self.mono = mono
        self.files = None

        self.get_file_list()
        self.convert()

    @staticmethod
    def mp3_to_wav(file):
        subprocess.call(['ffmpeg', '-i', file,
                         file.split('.mp3')[0] + '.wav'])

    @staticmethod
    def stereo_to_mono(file):
        sound = AudioSegment.from_wav(file.split('.mp3')[0] + '.wav')
        sound = sound.set_channels(1)
        sound.export(file.split('.mp3')[0] + ".wav", format="wav")

    def get_file_list(self):
        self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.direc) for f in filenames if
                      os.path.splitext(f)[1] == '.mp3']

    def convert(self):
        for file in self.files:
            logger.info(f'Converting {os.path.basename(file)} to wav...')
            self.mp3_to_wav(file)
            if self.mono:
                logger.info(f'Converting {os.path.basename(file)} to mono...')
                self.stereo_to_mono(file)


if __name__ == "__main__":
    ARGS = parse_input()
    ConvertAudio(ARGS.directory, ARGS.mono)
