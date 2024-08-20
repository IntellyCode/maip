import librosa.feature
import librosa.core as lc
from .util import generalise, scale, reduce_dimensions


def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extract and aggregate MFCC features from an audio signal.

    Parameters:
    - y: np.ndarray
        Audio time series.
    - sr: int
        Sampling rate of the audio.
    - n_mfcc: int
        Number of MFCC features to extract. Default is 13.

    Returns:
    - mfcc_aggregated: np.ndarray
        Aggregated MFCC features (mean, std, max, and min).
    """
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Concatenate these measures
    mfcc_aggregated = scale(generalise(mfcc))

    return mfcc_aggregated


def extract_mfcc_rd(y, sr, rd=4, n_mfcc=13):
    array = extract_mfcc(y, sr, n_mfcc=n_mfcc)
    return reduce_dimensions(array, rd)


if __name__ == '__main__':
    y, sr = lc.load("../../data/raw/classical/classical.00000.wav")
    mfcc = extract_mfcc(y, sr)
    print(mfcc)
    print(mfcc.shape)