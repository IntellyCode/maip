import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def generalise(array: np.ndarray,axis=1) ->np.ndarray:
    mean = array.mean(axis=axis)
    std = array.std(axis=axis)
    max_v = array.max(axis=axis)
    min_v = array.min(axis=axis)

    return np.concatenate([mean, std, max_v, min_v])


def scale(array: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(array.reshape(-1,1)).flatten()


def reduce_dimensions(features,n_components=4):
    """
    Apply PCA to reduce dimensionality of feature vectors to a single value.

    Parameters:
    - features: np.ndarray
        Array of feature vectors (shape (n_samples, 52)).

    Returns:
    - reduced_features: np.ndarray
        PCA-reduced features (shape (n_samples, 1)).
    """
    pca = PCA(n_components=n_components)  # Initialize PCA to reduce to 1 component
    reduced_features = pca.fit_transform(features)  # Fit and transform the features

    return reduced_features



