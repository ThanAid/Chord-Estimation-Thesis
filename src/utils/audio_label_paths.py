import os


def generate_output(audio_dir, labels_dir, shifts=False):
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
    # Define the paths to the audio and labels directories
    audio_dir = '/home/thanos/Documents/Thesis/audio/TheBeatlesShifted_Noise'
    labels_dir = '/home/thanos/Documents/Thesis/labels/TheBeatles_shifted'

    # Create or open the output text file
    output_file = '/home/thanos/Documents/Thesis/dataset_paths.txt'

    output_rows = generate_output(audio_dir, labels_dir, shifts=True)

    # Define the paths to the audio and labels directories
    audio_dir = '/home/thanos/Documents/Thesis/audio/TheBeatles'
    labels_dir = '/home/thanos/Documents/Thesis/labels/TheBeatles_lab'

    # Create or open the output text file
    output_file = '/home/thanos/Documents/Thesis/dataset_paths.txt'

    output_rows += generate_output(audio_dir, labels_dir, shifts=False)

    # Write the output rows to the text file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_rows))

    print(f"Output file '{output_file}' has been created.")
