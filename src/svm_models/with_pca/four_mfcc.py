"""
This loads a 52 element mfcc vector, then runs PCA to find the 4 most important elements of the vector and then trains
the SVM

Best Accuracy: 80% with pca_4
"""

from src.feature_extraction.mfcc import extract_mfcc
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions

features, labels = load(extract_mfcc)
features_2 = reduce_dimensions(features, 2)
features_3 = reduce_dimensions(features, 3)
features_4 = reduce_dimensions(features, 4)
features_5 = reduce_dimensions(features, 5)
train_svm(features_2, labels)
train_svm(features_3, labels)
train_svm(features_4, labels)
train_svm(features_5, labels)

