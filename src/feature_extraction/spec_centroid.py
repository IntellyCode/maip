import librosa.feature
import librosa.core as lc
from .util import generalise, scale


def extract_spectral_centroid(y, sr):
    """
    Extract and aggregate Spectral Centroid features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.

    Returns:
    - spec_centroid_aggregated: np.ndarray
        Aggregated Spectral Centroid features (mean, std, max, and min).
    """
    # Extract Spectral Centroid features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # Aggregate the features
    spec_centroid_aggregated = scale(generalise(spec_centroid))
    return spec_centroid_aggregated
