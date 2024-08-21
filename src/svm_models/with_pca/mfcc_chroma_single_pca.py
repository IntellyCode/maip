"""
Kernel SVM which uses 10 pca components

Accuracy at 10 pca: 78%
Accuracy at 50 pca: 82%
Accuracy at 80 pca: 82%
"""


from src.feature_extraction import mfcc, chroma
import numpy as np
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions


music, labels = load()
features = []
for m in music:
    mfcc_feature = mfcc.extract_mfcc(m[0],m[1])
    chroma_feature = chroma.extract_chroma(m[0],m[1])
    feature = np.concatenate((mfcc_feature, chroma_feature))
    features.append(feature)
features = reduce_dimensions(features, 80)

# print(features)
train_svm(features, labels)
