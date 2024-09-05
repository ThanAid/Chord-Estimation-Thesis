# Abstract

This thesis presents an innovative approach to audio chord recognition, aiming to
automatically identify and classify fundamental chord structures within music pieces.
Leveraging Convolutional Neural Networks (CNNs) with Bidirectional Long Short-Term
Memory (biLSTM) layers, advanced feature engineering, and post-processing techniques
rooted in music theory, our research enhances the accuracy and robustness of chord
recognition systems. By extracting features from chord representations such as root,
bass, and triad qualities, and segmenting the problem into distinct components, our
framework creates a solid ground to enhance the accuracy of chord recognition. Ad-
ditionally, we employ transfer learning techniques to capitalize on pre-trained models,
fine-tuning them for our specific chord recognition task, thus improving generalization
and robustness. Moreover, our exploration encompasses various Fourier transforms for
feature extraction, including Short-Time Fourier Transform (STFT) and Constant Q Trans-
form (CQT), to capture essential information from audio signals and optimize chord recog-
nition performance. Through extensive experimentation and evaluation of different CNN
and biLSTM configurations, as well as post-processing techniques, our approach demon-
strates significant enhancements in several aspectes of chord recognition. Overall, this
research contributes a comprehensive framework that leverages deep learning method-
ologies, sophisticated feature engineering, and post-processing techniques, showcasing
its potential to advance music information retrieval systems.

## Preprocessing

The preprocessing pipeline involves a series of steps to convert and transform the raw audio data into a format that can be used for further analysis or training. Below is a description of each preprocessing script and its purpose:

### 1. `convert_mp3_wav.py`
This script is responsible for converting `.mp3` files to `.wav` format, which is a standard format for audio processing tasks.

**Usage**:
```bash
python convert_mp3_wav.py --directory path/to/files --mono
```

* Converts MP3 audio files into WAV format for easier processing in subsequent steps.

### 2. `cqt_transform.py`
This script applies a Constant-Q Transform (CQT) to the audio data, which is useful for time-frequency analysis. CQT is a type of Fourier transform that provides higher frequency resolution at lower frequencies and is typically used in music and audio signal processing.

**Usage**:
```bash
python cqt_transform.py --paths_txt path/to/paths_txt --dest_txt path/to/destination_txt 
 ```

* You can use pooling by adding the `--pool` flag
* You can use `--noskip` to replace already transformed files
* Performs a CQT on audio files for frequency domain analysis.

### 3. `fourier_tr.py`
This script performs a standard Fourier transform on the audio data. The Fourier transform is useful for transforming the audio signal from the time domain into the frequency domain.

**Usage**:
```bash
python fourier_tr.py --paths_txt path/to/paths_txt --dest_txt path/to/destination_txt 
```

* You can use pooling by adding the `--pool` flag
* Transforms audio files using the Fourier Transform to obtain frequency components.

### 4. `pitch_shifter.py`
This script performs pitch shifting on the audio data, altering the pitch without affecting the tempo. Pitch shifting is commonly used for data augmentation in audio processing.

**Usage**:
```bash
python pitch_shifter.py --directory path/to/files --dest_dir path/to/destination --n_steps 5 --pool --noise
```
* You can optionally use `pool` and `noise` if you want to use pooling for parallel processing and to add Gaussian noise accordingly.
* Shifts the pitch of audio files for augmentation or other tasks.

### 5. `pitch_shift_labels.py`
This script is similar to `pitch_shifter.py`, but it also adjusts the corresponding labels for the shifted audio. This ensures that the labels remain consistent with the pitch-shifted data.

**Usage**:
```bash
python pitch_shift_labels.py --directory path/to/files --dest_dir path/to/destination --n_steps 5 --pool
```
* You can optionally use `pool` if you want to use pooling for parallel processing.
* Shifts the pitch of audio files and updates the corresponding labels accordingly.

