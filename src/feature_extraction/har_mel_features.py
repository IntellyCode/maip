import numpy as np
import librosa
from .util import generalise, scale


def extract_harmonic_melodic(y, sr):
    """
    Extract and aggregate harmonic and melodic features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.

    Returns:
    - harmonic_melodic_aggregated: np.ndarray
        Aggregated harmonic and melodic features (mean, std, max, min).
    """
    harmonic, _ = librosa.effects.harmonic(y=y), librosa.effects.percussive(y=y)
    harmonic_aggregated = scale(generalise(harmonic))
    melodic_aggregated = scale(generalise(_))
    harmonic_melodic_aggregated = np.concatenate([harmonic_aggregated, melodic_aggregated])
    return harmonic_melodic_aggregated
