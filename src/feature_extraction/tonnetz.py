import librosa.feature
import librosa.core as lc
from .util import generalise, scale


def extract_tonnetz(y, sr):
    """
    Extract and aggregate Tonnetz features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.

    Returns:
    - tonnetz_aggregated: np.ndarray
        Aggregated Tonnetz features (mean, std, max, and min).
    """
    # Extract Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    # Aggregate the features
    tonnetz_aggregated = scale(generalise(tonnetz))
    return tonnetz_aggregated
