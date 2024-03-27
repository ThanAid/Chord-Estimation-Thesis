"""From different csv files (one for root, bass etc) assemble a final csv with all the columns."""
import pandas as pd


class AssembleChord:
    def __init__(self, prediction_folder: str, file_name: str, save_path: str = None):
        """For the constructor to work correctly the file hierarcy on the prediction_folder should be as follows:
        prediction_folder
        ├── root
        │   └── file_name.csv
        ├── bass
        │   └── file_name.csv
        ├── triad
        │   └── file_name.csv
        ├── extension_1
        │   └── file_name.csv
        └── extension_2
            └── file_name.csv
        """
        self.root_path = f'{prediction_folder}/root/{file_name}'
        self.bass_path = f'{prediction_folder}/bass/{file_name}'
        self.triad_path = f'{prediction_folder}/triad/{file_name}'
        self.extension_1_path = f'{prediction_folder}/extension_1/{file_name}'
        self.extension_2_path = f'{prediction_folder}/extension_2/{file_name}'

        if save_path:
            self.save_path = save_path

    def assemble(self):
        """Assemble the predictions into one csv file."""
        root = pd.read_csv(self.root_path)
        bass = pd.read_csv(self.bass_path)
        triad = pd.read_csv(self.triad_path)
        extension_1 = pd.read_csv(self.extension_1_path)
        extension_2 = pd.read_csv(self.extension_2_path)

        final_df = pd.concat([root, bass, triad, extension_1, extension_2], axis=1)
        final_df.to_csv(self.save_path, index=False)
        return final_df
