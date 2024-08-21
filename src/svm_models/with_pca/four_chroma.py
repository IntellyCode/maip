"""
This loads a 48 element chroma vector, then runs PCA to find the 4 most important elements of the vector and then trains
the SVM

Best Accuracy: 78% with pca for 12 components
Most efficient: 70% with pca for 3 components
"""

from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions
from src.feature_extraction.chroma import extract_chroma


music, labels = load()
features = []
for m in music:
    features.append(extract_chroma(m[0], m[1]))
features_2 = reduce_dimensions(features, 2)
features_3 = reduce_dimensions(features, 3)
features_4 = reduce_dimensions(features, 4)
features_5 = reduce_dimensions(features, 5)
features_12 = reduce_dimensions(features, 12)
train_svm(features_2, labels)
train_svm(features_3, labels)
train_svm(features_4, labels)
train_svm(features_5, labels)
train_svm(features_12, labels)

