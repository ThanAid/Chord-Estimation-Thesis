import numpy as np
import librosa
import soundfile as sf


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


def stf_transform(audio, hop_size=4410, win_size=8192):
    """Short-Time Fourier Transform (STFT) with specified parameters"""

    stft = librosa.stft(audio, hop_length=hop_size, n_fft=win_size)
    return stft
