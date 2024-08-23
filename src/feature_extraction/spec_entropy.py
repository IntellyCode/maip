import numpy as np
import librosa.feature
from .util import generalise, scale


def extract_spectral_entropy(y, sr):
    """
    Extract and aggregate spectral entropy features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.

    Returns:
    - spectral_entropy_aggregated: np.ndarray
        Aggregated spectral entropy features (mean, std, max, min).
    """
    # Compute the power spectral density
    S = np.abs(librosa.stft(y))**2
    entropy = -np.sum(S * np.log(S + np.finfo(float).eps), axis=0)
    spectral_entropy_aggregated = scale(generalise(entropy))
    return spectral_entropy_aggregated
