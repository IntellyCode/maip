from src.feature_fusion.Fuser import generalise
import numpy as np

array = np.array([[1, 2, 4],
                     [1, 5, 6]])
print(generalise(array, axis=1, number_of_statistics=6))
