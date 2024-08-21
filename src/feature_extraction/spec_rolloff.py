import librosa.feature
from .util import generalise, scale, generalise_minimal


def extract_spectral_rolloff(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    # Aggregate as mean, std, min, max
    return scale(generalise(rolloff))

