"""
For this script to work correctly, audio and label directories must have the same structure.
"""
import os


def generate_output(audio_dir, labels_dir, shifts=False):
    """
    Generate a list of paths to audio files and their corresponding label files.

    Args:
        audio_dir (str): The path to the directory containing audio files.
        labels_dir (str): The path to the directory containing label files.
        shifts (bool): If True, iterate through subdirectories in the audio directory.

    Returns:
        list: A list of strings, each string representing a pair of paths (audio file, label file).
    """
    # Create a list to store the rows for the output file
    output_rows = []

    if shifts:
        for shift in os.listdir(audio_dir):
            # Traverse the audio directory and generate the rows
            current_audio_dir = os.path.join(audio_dir, shift)
            current_label_dir = os.path.join(labels_dir, shift)
            iterate_albums(current_audio_dir, current_label_dir, output_rows)
    else:
        iterate_albums(audio_dir, labels_dir, output_rows)

    return output_rows


def iterate_albums(current_audio_dir, current_label_dir, output_rows):
    """
    Iterate through albums in the given audio and label directories and generate paths.

    Args:
        current_audio_dir (str): Path to the current audio directory.
        current_label_dir (str): Path to the current label directory.
        output_rows (list): List to store the generated paths.
    """
    for album in os.listdir(current_audio_dir):
        album_audio_path = os.path.join(current_audio_dir, album)
        album_labels_path = os.path.join(current_label_dir, album)

        if os.path.isdir(album_audio_path) and os.path.isdir(album_labels_path):
            for song_file in os.listdir(album_audio_path):
                song_path = os.path.join(album_audio_path, song_file)

                # Look for the corresponding CSV file in the labels directory
                csv_file = f"{song_file.split('.')[0]}.lab"
                label_path = os.path.join(album_labels_path, csv_file)

                if os.path.exists(label_path):
                    output_rows.append(f"{song_path} {label_path}")


if __name__ == "__main__":
    # Define the paths to the audio and labels directories for shifted data
    audio_dir_shifted = '/home/thanos/Documents/Thesis/audio/TheBeatlesShifted_Noise'
    labels_dir_shifted = '/home/thanos/Documents/Thesis/labels/TheBeatles_shifted'

    # Create or open the output text file
    output_file = '/home/thanos/Documents/Thesis/dataset_paths.txt'

    output_rows = generate_output(audio_dir_shifted, labels_dir_shifted, shifts=True)

    # Define the paths to the audio and labels directories for non-shifted data
    audio_dir_non_shifted = '/home/thanos/Documents/Thesis/audio/TheBeatles'
    labels_dir_non_shifted = '/home/thanos/Documents/Thesis/labels/TheBeatles_lab'

    output_rows += generate_output(audio_dir_non_shifted, labels_dir_non_shifted, shifts=False)

    # Write the output rows to the text file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_rows))

    print(f"Output file '{output_file}' has been created.")
