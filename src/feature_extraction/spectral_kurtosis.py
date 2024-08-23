import numpy as np
import librosa.feature
from .util import generalise, scale


def extract_spectral_kurtosis(y, sr):
    """
    Extract and aggregate spectral kurtosis features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.

    Returns:
    - spectral_kurtosis_aggregated: np.ndarray
        Aggregated spectral kurtosis features (mean, std, max, min).
    """
    # Compute the power spectral density
    S = np.abs(librosa.stft(y))**2
    mean_S = np.mean(S, axis=0)
    std_S = np.std(S, axis=0)
    kurtosis = np.mean(((S - mean_S)**4) / (std_S**4), axis=0) - 3
    spectral_kurtosis_aggregated = scale(generalise(kurtosis))
    return spectral_kurtosis_aggregated
