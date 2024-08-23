import numpy as np
import librosa.beat
from .util import generalise, scale


def extract_beat_positions(y, sr):
    """
    Extract and aggregate beat positions features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.

    Returns:
    - beat_positions_aggregated: np.ndarray
        Aggregated beat positions features (mean, std, max, min).
    """
    beats = librosa.beat.beat_track(y=y, sr=sr)[1]
    beat_positions = np.diff(beats)
    beat_positions_aggregated = scale(generalise(beat_positions))
    return beat_positions_aggregated
