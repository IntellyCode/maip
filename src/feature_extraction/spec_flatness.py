import librosa.feature
import librosa.core as lc
from .util import generalise, scale


def extract_spectral_flatness(y, sr):
    """
    Extract and aggregate Spectral Flatness features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.

    Returns:
    - spec_flatness_aggregated: np.ndarray
        Aggregated Spectral Flatness features (mean, std, max, and min).
    """
    # Extract Spectral Flatness features
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    # Aggregate the features
    spec_flatness_aggregated = scale(generalise(spec_flatness))
    return spec_flatness_aggregated
