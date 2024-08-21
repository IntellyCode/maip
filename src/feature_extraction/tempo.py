import librosa
import numpy as np


def extract_tempo(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    return np.array(tempo)  # Single value representing the tempo
