import numpy as np
import librosa
import soundfile as sf
import pandas as pd


def add_noise(y):
    """adds Gaussian noise to a audiofile"""

    mean = np.mean(y)
    var = np.var(y)
    noise = np.random.normal(mean, var, y.shape)
    return y + noise


def load_wav(filepath, sampling_rate=44100):
    """loads a wav file"""

    y, sr = librosa.load(filepath, sr=sampling_rate)
    return y, sr


def export_wav(filepath, y, sampling_rate=44100):
    """Exports a file as wav"""

    sf.write(filepath, data=y, samplerate=sampling_rate)


def pitch_shift(y, sr=44100, n_steps=1):
    """Given a wav this function pitch shifts it"""

    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y_shifted


def stf_transform(audio, hop_size=4410, win_size=8192, q=True):
    """
    Short-Time Fourier Transform (STFT) with specified parameters

    :param audio: audio file in a waveform format
    :param hop_size:
    :param win_size:
    :param q: if True performs Constant Q Transform chromagram to calculate Pitch Class Profile (PCP), normalized
    :return:
    """

    if q:
        # Harmonic content extraction
        y_harmonic, y_percussive = librosa.effects.hpss(audio)

        # use Constant Q Transform to calculate Pitch Class Profile (PCP), normalized
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=44100, hop_length=hop_size)
        return chromagram.T
    else:
        stft = librosa.stft(audio, hop_length=hop_size, n_fft=win_size)
        return stft


def cqt_spectogram(audio, sample_rate, nbins, bins_per_octave, hop_length):
    """
    Performs CQT transform to an audio file

    :param audio: audio file in a waveform format
    :param sample_rate:
    :param nbins:
    :param bins_per_octave:
    :param hop_length:
    :return:
    """

    track_time = librosa.get_duration(y=audio, sr=sample_rate)
    spectrogram = librosa.cqt(audio, sr=sample_rate, n_bins=nbins, bins_per_octave=bins_per_octave,
                              hop_length=hop_length)

    frames = list(range(0, spectrogram.shape[1]))
    times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)

    timeseries = librosa.amplitude_to_db(abs(spectrogram), ref=np.max).T
    return timeseries, times, track_time


def read_transformed_audio(audio_path):
    """Reads csv containing fourier of audio"""

    audio = pd.read_csv(audio_path, header=None, sep=',')
    return audio
