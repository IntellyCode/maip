import librosa
import librosa.feature
import numpy as np


def extract_tempo(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(y=y, sr=sr, onset_envelope=onset_env)
    return np.array(tempo)  # Single value representing the tempo
