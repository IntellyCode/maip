import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _preprocessor(var_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if getattr(self, var_name) is None:
                raise ValueError(f"{var_name} is None.")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def _check_fusion(x: np.ndarray, y: np.ndarray):
    if len(x) != len(y):
        raise ValueError("Features must have same length")


class Fuser:
    def __init__(self):
        self._features = np.array([])
        self._loaded = np.array([])
        self._pca = None

    def _set_pca(self, num_components=2):
        self._pca = PCA(n_components=num_components)

    def run_pca(self, n_components=4):
        self._set_pca(num_components=n_components)
        return self._pca.fit_transform(self._loaded)

    def load(self, new_features: np.ndarray):
        if self._loaded:
            _check_fusion(self._loaded, new_features)
            self.fuse()
            self._loaded = new_features
        else:
            self._loaded = new_features

    def fuse(self):
        if self._features:
            _check_fusion(self._features,self._loaded)
            self._features = [np.concatenate((self._features[i], self._loaded[i])) for i in range(len(self._features))]
            self._loaded = np.array([])
        else:
            self._features = self._loaded

    def get_features(self):
        return self._features


