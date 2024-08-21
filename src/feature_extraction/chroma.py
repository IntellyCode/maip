import librosa.feature
from .util import generalise, scale, reduce_dimensions


def extract_chroma(y, sr, n_chroma=12):
    """
    Extract Chroma features and summarize them into a single feature vector.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.
    - n_chroma: int
        Number of Chroma features. Default is 12.

    Returns:
    - chroma_features: np.ndarray
        Concatenated feature vector summarizing the Chroma features.
    """
    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)

    # Concatenate these measures into a single feature vector
    chroma_features = scale(generalise(chroma))

    return chroma_features



