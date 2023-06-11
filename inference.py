from scipy.io import wavfile
from tensorflow import keras
import numpy as np
from scipy.signal import spectrogram
import librosa
from processing import SpectrogramProcessor

TIME_FRAMES = 12


def read_wav_file(x):
    # Read wavfile using scipy wavfile.read.
    _, wav = wavfile.read(x)

    # Normalize.
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max

    wav_dim = np.shape(wav)
    if len(wav_dim) == 2:
        # Convert stereo to mono.
        wav = wav.sum(axis=1) / 2

    return wav


def get_predictions(wav_file: str, model_path: str):
    win_length = 2048
    annotation_file = ""
    sample_rate = 44100
    mel_bands = 80
    threshold_freq = 15500

    wav = read_wav_file(wav_file)
    hop_length = 441  # win_length // 4 # 2048 // 4 = 512
    noverlap = win_length - hop_length

    freqs, times, spec = spectrogram(
        wav,
        sample_rate,
        window="hann",
        nperseg=win_length,
        noverlap=noverlap,
        mode="complex",
    )
    _, S_percussive = librosa.decompose.hpss(spec, margin=(1.0, 5.0))
    S = librosa.feature.melspectrogram(
        S=np.abs(S_percussive),
        sr=sample_rate,
        window="hann",
        win_length=win_length,
        hop_length=hop_length,
        n_mels=mel_bands,
        center=False,
        fmax=threshold_freq,
    )
    S_db = librosa.core.power_to_db(S, ref=np.max)
    S_expanded = np.expand_dims(S_db, axis=2)

    spectrograms = []
    sp = SpectrogramProcessor(S_expanded, times, annotation_file)
    spectrograms = sp.split_spectrogram(TIME_FRAMES)
    spectrograms = [s[0] for s in spectrograms]
    spectrograms = np.array(spectrograms)

    model = keras.models.load_model(model_path)
    y_pred = model.predict(spectrograms, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred
