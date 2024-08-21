import librosa.feature
import librosa.core as lc
from .util import generalise, scale


def extract_mel_spectrogram(y, sr, n_mels=128):
    """
    Extract and aggregate Mel-Spectrogram features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.
    - n_mels: int
        Number of Mel bands to generate. Default is 128.

    Returns:
    - mel_spec_aggregated: np.ndarray
        Aggregated Mel-Spectrogram features (mean, std, max, and min).
    """
    # Extract Mel-Spectrogram features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    # Aggregate the features
    mel_spec_aggregated = scale(generalise(mel_spec))
    return mel_spec_aggregated
