import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import Sequence
from collections import Counter

from src.adapt_labels import *
from src.utils import label_utils
from src.utils.audio_utils import *


class DataGenerator(Sequence):
    """
    Generator for batches in TF

    Example of use:

    train_gen = DataGenerator(X_train, y_train, 32)
    test_gen = DataGenerator(X_test, y_test, 32)


    history = model.fit(train_gen,
                        epochs=6,
                        validation_data=test_gen)
    """

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def slice_freeze_model(model, n_sliced_layers=3):
    """
    given a tf keras model and a number of layers to slice (from the end) it returns a sliced model usable for transfer
    learning. If freeze is True it also freezes parameters from conv layers.

    :param model: keras model
    :param n_sliced_layers: (int) number of layers to cut
    :return: keras model
    """

    # remove the last layers (n sclided)
    sliced_model = tf.keras.Sequential(model.layers[:-n_sliced_layers])

    # Freeze, set trainable=False for the layers from loaded_model
    for layer in sliced_model.layers:
        layer.trainable = False

    return sliced_model


def plot_cm(model, X_test, y_test, label_mapping, is_2d=False):
    """
    Plots confusion matrix
    :param model: keras model
    :param X_test: array like
    :param y_test: encoded y test
    :param label_mapping: dict with encoding map
    :return:
    """
    y_pred = model.predict(X_test)
    if is_2d:
        # Get the actual predictions
        y_pred_id = np.hstack([np.argmax(y, axis=1) for y in y_pred])
        y_test = np.hstack([np.argmax(y, axis=1) for y in y_test])
    else:
        y_pred_id = [np.argmax(y, axis=1) for y in y_pred]

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_id)

    # Get the labels in the correct order
    labels = list(label_mapping.keys())

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


class DataChunking:
    """
    Creates chunks using the data given in order to have 2-d data chunks.
    """

    def __init__(self, paths_txt, dest_file, chunk_size, label_col='root', dataframe=False, verbose=200, y_only=False,
                 encoder=None, encoding_dict=None):
        """

        :param paths_txt: txt or dataframe containing paths, if dataframe the dataframe must be set True
        :param dest_file: dest file path to save the data
        :param chunk_size: (int) size for each chunk
        :param label_col: the label of the column wanted to predict, for example 'root' or 'bass'
        :param dataframe: if True then paths_txt is already a dataframe, if False it is a txt and needs to be read
        """
        if not dataframe:
            self.paths_df = pd.read_csv(paths_txt, delimiter=' ', index_col=False, names=['wav', 'labels'], header=None,
                                        low_memory=False)
        else:
            self.paths_df = paths_txt
        self.dest_file = dest_file
        self.pooling = False
        self.trsn_paths_df = pd.DataFrame(columns=['audio_csv', 'labels'])
        self.label_col = label_col

        self.chunk_size = chunk_size
        self.verbose = verbose

        # The following will be initialized at the first iteration
        self.input_features = None
        self.label_col_size = None  # if OHE is used more than a column needed for labels

        self.X, self.y = None, None
        self.initialized = False

        # Encoding and encoder for labels
        self.encoder = encoder
        self.encoding_dict = encoding_dict

        self.y_only = y_only

    def init_arrays(self):
        """
        Initialize X_train, y_train, X_test, y_test

        :return:
        """
        self.X = np.zeros((1, self.chunk_size, self.input_features))  # num of frequencies
        self.y = np.zeros((1, self.chunk_size, self.label_col_size))

        self.initialized = True
        return self

    def encode_lbls(self, annotations, n_feat):
        """

        :param annotations:
        :param n_feat:
        :return:
        """
        encoded_lbls = np.empty((n_feat,))
        for row in annotations:
            encoded_lbls = np.column_stack((encoded_lbls, self.encoder.transform(row.reshape(-1, 1)).toarray()[0]))
        return encoded_lbls

    def read_data(self, row):
        audio_path = row['wav']
        lbl_path = row['labels']

        # Assuming you have a function to read audio and label data, replace the placeholders below
        timeseries = read_transformed_audio(audio_path).to_numpy()

        # Read label column
        label_df = pd.read_csv(lbl_path, header=None, sep=' ')
        # Extract features from chord
        y_train_features = ConvertLab(label_df, label_col=1, dest=None, is_df=True)

        # Encoding y data
        annotations = y_train_features.df[self.label_col].values
        annotations = np.vectorize(self.encoding_dict.get)(annotations)

        # encoded_annotations = self.encode_lbls(annotations, n_feat=len(self.encoder.categories_[0])).T
        annotations = self.encoder.transform(annotations.reshape(-1, 1)).A

        return timeseries, annotations

    def chunkify(self):
        """
        Reads data creates chunks and appends the data to self.X_train etc...

        :return:
        """
        logger.info("Started Chunking...")
        for i, (index, row) in enumerate(self.paths_df.iterrows()):
            # Read row
            timeseries, annotations = self.read_data(row)
            # annotations = annotations.T

            # The first iter, data is initialized
            if not self.initialized:
                self.input_features = timeseries.shape[1]
                try:
                    self.label_col_size = annotations.shape[1]
                except IndexError:
                    logger.warning("Index error on checking the number of columns in annotation, resulting to 1.")
                    # In this case there is only one column
                    self.label_col_size = 1

                # Initialize arrays for data
                self.init_arrays()

            timestep = 0
            # size of the current track
            chunks = len(timeseries)

            # slice and stack train
            while timestep < chunks:
                if (chunks - timestep) > self.chunk_size:
                    # X side
                    if not self.y_only:
                        batch_x = np.resize(timeseries[timestep:timestep + self.chunk_size, :],
                                            (1, self.chunk_size, self.input_features))  # num of frequencies
                        self.X = np.append(self.X, batch_x, axis=0)

                    # y side
                    batch_y = np.resize(annotations[timestep:timestep + self.chunk_size],
                                        (1, self.chunk_size, self.label_col_size))
                    self.y = np.append(self.y, batch_y, axis=0)

                # If there is no enough row left for a whole chunk then we pad
                else:
                    if not self.y_only:
                        batch_x = timeseries[timestep:, :]
                    batch_y = annotations[timestep:]
                    for step in range(0, self.chunk_size + timestep - chunks):
                        if not self.y_only:
                            batch_x = np.vstack((batch_x, np.zeros((1, self.input_features))))  # fill with zeros
                        batch_y = np.append(batch_y, self.encoder.transform(np.array(0).reshape(-1, 1)).A)

                    if not self.y_only:
                        self.X = np.append(self.X, np.array([batch_x]), axis=0)
                    batch_y = np.resize(batch_y, (1, self.chunk_size, self.label_col_size))
                    self.y = np.append(self.y, batch_y, axis=0)

                timestep += self.chunk_size

            if self.verbose != 0:
                if i % self.verbose == 0:
                    logger.info(f"{i} iterations completed.")

        return self

    def delete_first_chunk(self):
        """
        First chunks are zero from the initialization, so they need to be deleted.

        :return:
        """
        if not self.y_only:
            self.X = np.delete(self.X, 0, 0)

        self.y = np.delete(self.y, 0, 0)

        return self

    def get_data(self):
        """

        :return: X and y data (numpy format)
        """

        return self.X, self.y

    def run_chunkify(self):
        """
        Runs the pipe
        :return:
        """
        self.chunkify().delete_first_chunk()
        return self

    def get_weights(self):
        """
        Calculates the weight for each class in order to balance data.

        :return: (dict) keys are classes values are weights.
        """

        weights = dict(Counter([y.argmax() for chunk in self.y for y in chunk]))
        n_samples = sum(weights.values())
        for k in weights:
            # formula for class weights
            weights[k] = n_samples / (len(weights) * weights[k])
        return weights
