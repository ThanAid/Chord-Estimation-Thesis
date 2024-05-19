from tensorflow.keras.utils import plot_model
import tensorflow as tf

model = tf.keras.models.load_model("/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/deep_cnn/models/CQT_cnn_root_10.h5")

model.summary()

plot_model(model, to_file="/home/thanos/Documents/Thesis/model_plots/model_2_root.png", show_shapes=True)
