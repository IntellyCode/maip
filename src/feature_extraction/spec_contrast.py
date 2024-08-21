import librosa.feature
from .util import generalise, scale, generalise_minimal


def extract_spectral_contrast(y, sr):
    # extracts the 7 spectral_contrast features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    return scale(generalise_minimal(spectral_contrast))

