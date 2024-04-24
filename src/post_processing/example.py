from src.post_processing.smoothing import smooth_column, chord_filter
import pandas as pd

if __name__ == "__main__":
    input = pd.read_csv("/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/predictions/assembled/y_pred_TheBeatles_01_-_Please_Please_Me_01_-_I_Saw_Her_Standing_There_CQT.csv")
    actual = pd.read_csv("/home/thanos/Documents/Thesis/Chord-Estimation-Thesis/src/2d_cnn/predictions/assembled/y_TheBeatles_01_-_Please_Please_Me_01_-_I_Saw_Her_Standing_There_CQT.csv")

    from src.metrics.mirex import chord_parts_precision

    out2 = smooth_column(input, "0.1", window_size=5, in_place=False)
    out2 = smooth_column(out2, "0", window_size=5, in_place=True)
    out2 = smooth_column(out2, "0.2", window_size=5, in_place=True)

    output = chord_filter(out2)

    print("Mirex b4 smoothing: ", chord_parts_precision(actual, input))
    print("Mirex after smoothing: ", chord_parts_precision(actual, out2))
    print("Mirex after smoothing & filtering: ", chord_parts_precision(actual, output))

    print("Mirex after: ", chord_parts_precision(actual, out2))