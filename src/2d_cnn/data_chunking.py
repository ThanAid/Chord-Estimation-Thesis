
from src.utils.audio_utils import *
from src.adapt_labels import *

chunk_size = 100
input_features = 192
# train set
x_train = np.zeros((1, chunk_size, input_features))  # num of frequencies
y_train_root = np.zeros((1, chunk_size, 1))  # TODO: Change to (1, chunk_size, 14) if use onehotencoder
y_train_bass = np.zeros((1, chunk_size, 1))
# test set
x_test = np.zeros((1, chunk_size, input_features))  # num of frequencies
y_test_root = np.zeros((1, chunk_size, 1))
y_test_bass = np.zeros((1, chunk_size, 1))

file_path = '/home/thanos/Documents/Thesis/Dataset_paths/dataset_paths_CQT.txt'

df = pd.read_csv(file_path, delimiter=' ', index_col=False, names=['wav', 'labels'], header=None, low_memory=False)

# Iterate through rows and concatenate audio and label data
for i, (index, row) in enumerate(df.iterrows()):
    audio_path = row['wav']
    label_path = row['labels']

    # Assuming you have a function to read audio and label data, replace the placeholders below
    timeseries = read_transformed_audio(audio_path).to_numpy()

    # Read label column
    label_df = pd.read_csv(label_path, header=None, sep=' ')
    # Extract features from chord
    y_train_features = ConvertLab(label_df, label_col=1, dest=None, is_df=True)
    y_root = y_train_features.df['root'].values
    y_bass = y_train_features.df['bass'].values

    # for album in root_vec.keys():
    #     for track_no in root_vec[album].keys():
    # if len([(tr,alb) for tr, alb in X_tracks_test if album.find(alb) and tr == track_no]) > 0:
    #     continue
    timestep = 0
    # size of the current track
    chunks = len(timeseries)

    # track annotations
    annotations_root = y_root.T
    annotations_bass = y_bass.T
    # slice and stack train
    while timestep < chunks:
        if (chunks - timestep) > chunk_size:
            batch_x = np.resize(timeseries[timestep:timestep + chunk_size, :],
                                (1, chunk_size, input_features))  # num of frequencies
            x_train = np.append(x_train, batch_x, axis=0)
            batch_y = np.resize(annotations_root[timestep:timestep + chunk_size], (1, chunk_size, 1))
            # TODO: Change to (1, chunk_size, 14) if use onehotencoder
            y_train_root = np.append(y_train_root, batch_y, axis=0)
            batch_y = np.resize(annotations_bass[timestep:timestep + chunk_size], (1, chunk_size, 1))
            y_train_bass = np.append(y_train_bass, batch_y, axis=0)
        else:
            batch_x = timeseries[timestep:, :]
            batch_y_root = annotations_root[timestep:]
            batch_y_bass = annotations_bass[timestep:]
            for step in range(0, chunk_size + timestep - chunks):
                batch_x = np.vstack((batch_x, np.zeros((1, input_features))))
                # batch_y_root = np.vstack((batch_y_root, encoder.transform([[df.loc['N']['Root']]]).toarray()[0]))
                batch_y_root = np.append(batch_y_root, 'N')
                # batch_y_bass = np.vstack((batch_y_bass, encoder.transform([[df.loc['N']['Root']]]).toarray()[0]))
                batch_y_bass = np.append(batch_y_bass, 'N')
            x_train = np.append(x_train, np.array([batch_x]), axis=0)
            batch_y_root = np.resize(batch_y_root, (1, chunk_size, 1))
            y_train_root = np.append(y_train_root, batch_y_root, axis=0)
            batch_y_bass = np.resize(batch_y_bass, (1, chunk_size, 1))
            y_train_bass = np.append(y_train_bass, batch_y_bass, axis=0)
        timestep += chunk_size
    break


# Delete the first row from every array because of the append, which left it all zeros.

x_train = np.delete(x_train,0,0)
x_test = np.delete(x_test,0,0)

y_train_root = np.delete(y_train_root,0,0)
y_test_root = np.delete(y_test_root,0,0)

y_train_bass = np.delete(y_train_bass,0,0)
y_test_bass = np.delete(y_test_bass,0,0)

y_train_quality = np.delete(y_train_quality,0,0)
y_test_quality = np.delete(y_test_quality,0,0)

print(x_train.shape, y_train_root.shape, y_train_bass.shape, x_test.shape, y_test_root.shape, y_test_bass.shape)
