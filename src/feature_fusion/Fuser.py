import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


_HashMap = {
    "mean": np.ndarray.mean,
    "std": np.ndarray.std,
    "min": np.ndarray.min,
    "max": np.ndarray.max,
    "skewness": scipy.stats.skew,
    "kurtosis": scipy.stats.kurtosis,
}
_List = ["mean", "std", "min", "max", "skewness", "kurtosis"]


def generalise(array: np.ndarray, axis=1, number_of_statistics=4) -> np.ndarray:
    if number_of_statistics < 1 or number_of_statistics > 6:
        raise ValueError("Number of statistics must be between 1 and 6")
    stats = []
    for i in range(number_of_statistics):
        key = _List[i]
        func = _HashMap[key]
        stats.append(func(array, axis=axis))
    return np.concatenate([i for i in stats])


def scale(array: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(array.reshape(-1, 1)).flatten()


def _preprocessor(var_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if getattr(self, var_name) is None:
                raise ValueError(f"{var_name} is None.")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class Fuser:
    def __init__(self):
        self.features = []
        self.pca = None

    def _set_pca(self, num_components=2):
        self.pca = PCA(n_components=num_components)

    def run_pca(self, features, n_components=4):
        self._set_pca(num_components=n_components)
        return self.pca.fit_transform(features)

    def fuse(self,new_features):
        if self.features:
            pass


            

