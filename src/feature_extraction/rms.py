import librosa.feature
from .util import generalise, scale, generalise_minimal


def extract_rms(y):
    rms = librosa.feature.rms(y=y)
    # Aggregate as mean, std, min, max
    return scale(generalise(rms))
