from tensorflow.keras.utils import plot_model
import tensorflow as tf

model = tf.keras.models.load_model("/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/naive_cnn/naive_cnn.h5")

model.summary()

plot_model(model, to_file="/home/thanos/Documents/Thesis/model_plots/naive_cnn.png", show_shapes=True)