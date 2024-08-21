"""
This loads a 52 element mfcc vector, then runs PCA to find the 4 most important elements of the vector and then trains
the SVM

Best Accuracy: 80% with pca_4
"""

from src.feature_extraction.mfcc import extract_mfcc
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions

music, labels = load()
features = []
for m in music:
    features.append(extract_mfcc(m[0], m[1]))
features_4 = reduce_dimensions(features, 4)
train_svm(features_4, labels)

