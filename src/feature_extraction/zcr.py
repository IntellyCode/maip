import librosa.feature
from .util import generalise, scale, generalise_minimal


def extract_zero_crossing_rate(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    # Aggregate as mean, std, min, max
    return scale(generalise(zcr))
