import librosa.feature
from .util import generalise, scale, generalise_minimal


def extract_spectral_bandwidth(y, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # Aggregate as mean, std, min, max
    return scale(generalise(bandwidth))
