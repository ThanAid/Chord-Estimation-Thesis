import gc
import pickle
import sys

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam

sys.path.append("../src")

# import local libs
from src.utils.train_utils import *


class TransferLearning:
    """
        A class for transfer learning with a pre-trained model for audio classification tasks.

        Args:
            model_path (str): Path to the pre-trained model file.
            label_col (str): Column name containing the labels in the dataset.
            cache_path (str): Path to the directory where cached data is stored.
            dest_model (str): Path to save the trained transfer learning model.
            n_sliced_layers (int, optional): Number of layers to slice and freeze from the pre-trained model.
                Defaults to 2.
            batch_size (int, optional): Batch size for training and testing data generators. Defaults to 32.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            epochs (int, optional): Number of epochs for model training. Defaults to 10.
    """

    def __init__(self, model_path, label_col, cache_path, dest_model, n_sliced_layers=2, batch_size=32, lr=0.001,
                 epochs=10):
        self.model_path = model_path
        self.label_col = label_col
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dest_model = dest_model
        self.n_sliced_layers = n_sliced_layers
        self.model = None
        self.X_test = None
        self.test_gen = None
        self.X_train = None
        self.train_gen = None
        self.y_train = None
        self.y_test = None
        self.y_test_encoded = None
        self.y_train_encoded = None
        self.label_mapping = None

    def load_cut_freeze(self):
        """
        Load the pre-trained model, slice and freeze specified layers, and store the modified model.

        :return:
        """
        logger.info("Loading the model..")
        model = tf.keras.models.load_model(self.model_path)

        logger.info(model.summary())

        logger.info("Slicing and freezing model")
        self.model = slice_freeze_model(model, n_sliced_layers=self.n_sliced_layers)
        logger.info(self.model.summary())
        return self

    def extract_features_labels(self):
        """
        Load data from cache, extract label features, and prepare training and testing data.

        :return:
        """
        logger.info("Loading data from cache and extracting label features...")
        self.X_train = np.loadtxt(f"{self.cache_path}/X_train.csv", delimiter=",")
        self.X_test = np.loadtxt(f"{self.cache_path}/X_test.csv", delimiter=",")
        df_y_train = pd.read_csv(f"{self.cache_path}/y_train.csv", delimiter=",", low_memory=False,
                                 index_col=False)
        df_y_test = pd.read_csv(f"{self.cache_path}/y_test.csv", delimiter=",", low_memory=False,
                                index_col=False)

        # Split labels into features
        logger.info("Splitting labels into features..")
        y_train_features = ConvertLab(df_y_train, label_col='labels', dest=None, is_df=True)
        self.y_train = y_train_features.df[self.label_col].values

        y_test_features = ConvertLab(df_y_test, label_col='labels', dest=None, is_df=True)
        self.y_test = y_test_features.df[self.label_col].values

        return self

    def encode_labels(self):
        """
        Encode the labels using LabelEncoder and store the label mapping.

        :return:
        """
        logger.info("Encoding labels...")
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Fit and transform all the labels
        y_encoded = label_encoder.fit_transform(np.concatenate((self.y_test, self.y_train)))

        # Transform the train and test labels using the same encoder
        self.y_train_encoded = label_encoder.transform(self.y_train)
        self.y_test_encoded = label_encoder.transform(self.y_test)

        # Now y_train_encoded and y_test_encoded contain the encoded labels

        # To get the mapping from original labels to encoded labels, you can use:
        self.label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        return self

    def reshape_data(self):
        """
         Reshape the data for CNN input and create data generators for training and testing.

        :return:
        """
        logger.info("Reshaping Data and mounting generators...")
        # Reshape the features for CNN input
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)

        # Using generators to deal with running out of memory on GPU
        self.train_gen = DataGenerator(self.X_train, self.y_train_encoded, self.batch_size)
        self.test_gen = DataGenerator(self.X_test, self.y_test_encoded, self.batch_size)

        # Delete X train as we no longer need them
        del self.X_train

        gc.collect()

        return self

    def add_layers(self):
        """
        Add trainable layers to the model.

        """
        logger.info("Adding new layers...")
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(len(np.unique(np.concatenate((self.y_test_encoded, self.y_train_encoded))))
                                             , activation='softmax'))
        logger.info("Model after adding layers:\n", self.model.summary())

        return self

    def train(self):
        """
         Compile and train the transfer learning model.

        :return:
        """
        logger.info("Training starting up...")
        # Print shapes using print statements
        for layer in self.model.layers:
            print(layer.name, layer.output_shape)

        optimizer = Adam(learning_rate=0.0001)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(self.train_gen, epochs=self.epochs, validation_data=self.test_gen)
        return self

    def save_model(self):
        """
         Save the trained transfer learning model if a destination path is provided.

        :return:
        """
        if self.dest_model:
            logger.info(f"Saving model to {self.dest_model}.")
            self.model.save(self.dest_model)

        return self

    def plot_confusion_m(self):
        """
        Plot the confusion matrix for the trained model on the test data.

        :return:
        """
        plot_cm(self.model, self.X_test, self.y_test_encoded, self.label_mapping)

        return self

    def run_pipeline(self):
        """
         Run the complete transfer learning pipeline.

        :return:
        """
        (self.load_cut_freeze().extract_features_labels().encode_labels().reshape_data().add_layers().train().
         save_model().plot_confusion_m())


class TransferLearning2D:
    """
        A class for transfer learning with a pre-trained model for audio classification tasks for 2d model.

        Args:
            model_path (str): Path to the pre-trained model file.
            label_col (str): Column name containing the labels in the dataset.
            cache_path (str): Path to the directory where cached data is stored.
            dest_model (str): Path to save the trained transfer learning model.
            n_sliced_layers (int, optional): Number of layers to slice and freeze from the pre-trained model.
                Defaults to 2.
            batch_size (int, optional): Batch size for training and testing data generators. Defaults to 32.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            epochs (int, optional): Number of epochs for model training. Defaults to 10.
    """

    def __init__(self, model_path, label_col, cache_path, dest_model, n_sliced_layers=2, batch_size=32, lr=0.001,
                 epochs=10, label_mapping=None, use_weights=False):
        self.model_path = model_path
        self.label_col = label_col
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dest_model = dest_model
        self.n_sliced_layers = n_sliced_layers
        self.model = None
        self.X_test = None
        self.test_gen = None
        self.X_train = None
        self.train_gen = None
        self.y_test_encoded = None
        self.y_train_encoded = None
        self.y_weights = None
        self.label_mapping = label_mapping
        self.use_weights = use_weights

    def load_cut_freeze(self):
        """
        Load the pre-trained model, slice and freeze specified layers, and store the modified model.

        :return:
        """
        logger.info("Loading the model..")
        model = tf.keras.models.load_model(self.model_path)

        logger.info(model.summary())

        logger.info("Slicing and freezing model")
        self.model = slice_freeze_model(model, n_sliced_layers=self.n_sliced_layers)
        logger.info(self.model.summary())
        return self

    def read_data(self):
        """
        Load data from cache.

        :return:
        """
        with open(f"{self.cache_path}/X_train.pickle", 'rb') as f:
            self.X_train = pickle.load(f)

        with open(f"{self.cache_path}/X_test.pickle", 'rb') as f:
            self.X_test = pickle.load(f)

        with open(f"{self.cache_path}/y_train_{self.label_col}.pickle", 'rb') as f:
            self.y_train_encoded = pickle.load(f)

        with open(f"{self.cache_path}/y_test_{self.label_col}.pickle", 'rb') as f:
            self.y_test_encoded = pickle.load(f)

        with open(f"{self.cache_path}/y_train_weights_{self.label_col}.pickle", 'rb') as f:
            self.y_weights = pickle.load(f)

        return self

    def reshape_data(self):
        """
         Reshape the data for CNN input and create data generators for training and testing.

        :return:
        """
        logger.info("Reshaping Data and mounting generators...")
        # Reshape the features for CNN input
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1)

        # y data is already encoded
        # Using generators to deal with running out of memory on GPU
        self.train_gen = DataGenerator(self.X_train, self.y_train_encoded, self.batch_size)
        self.test_gen = DataGenerator(self.X_test, self.y_test_encoded, self.batch_size)

        # Delete X train as we no longer need them
        del self.X_train

        gc.collect()

        return self

    def add_layers(self):
        """
        Add trainable layers to the model.

        """
        logger.info("Adding new layers...")
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True),
                                                     name='LSTM_layer'))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, return_sequences=True),
                                                     name='LSTM_layer2'))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(self.y_weights), activation='softmax'),
                                                       name='out'))
        logger.info("Model after adding layers:\n", self.model.summary())

        return self

    def train(self):
        """
         Compile and train the transfer learning model.

        :return:
        """
        logger.info("Training starting up...")
        # Print shapes using print statements
        for layer in self.model.layers:
            print(layer.name, layer.output_shape)

        optimizer = Adam(learning_rate=0.0001)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        if self.use_weights:
            self.model.fit(self.train_gen, epochs=self.epochs, validation_data=self.test_gen, class_weight=self.y_weights)
        else:
            self.model.fit(self.train_gen, epochs=self.epochs, validation_data=self.test_gen)
        return self

    def save_model(self):
        """
         Save the trained transfer learning model if a destination path is provided.

        :return:
        """
        if self.dest_model:
            logger.info(f"Saving model to {self.dest_model}.")
            self.model.save(self.dest_model)

        return self

    def plot_confusion_m(self):
        """
        Plot the confusion matrix for the trained model on the test data.

        :return:
        """
        plot_cm(self.model, self.X_test, self.y_test_encoded, self.label_mapping, is_2d=True)

        return self

    def run_pipeline(self):
        """
         Run the complete transfer learning pipeline.

        :return:
        """
        (self.load_cut_freeze().read_data().reshape_data().add_layers().train().
         save_model().plot_confusion_m())
