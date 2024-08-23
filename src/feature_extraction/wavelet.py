import pywt
import numpy as np
from .util import generalise, scale


def extract_wavelet_transform(y, sr, wavelet='haar'):
    """
    Extract and aggregate wavelet transform features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.
    - wavelet: str
        Type of wavelet to use.

    Returns:
    - wavelet_aggregated: np.ndarray
        Aggregated wavelet transform features (mean, std, max, min).
    """
    coeffs = pywt.wavedec(y, wavelet)
    # Flatten coefficients and aggregate
    coeffs_flat = np.concatenate([c.flatten() for c in coeffs])
    wavelet_aggregated = scale(generalise(coeffs_flat))
    return wavelet_aggregated
