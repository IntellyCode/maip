"""
Kernel SVM which uses 4 components of the mfcc and 3 components of the chrome to classify music genres

Accuracy: 78%
"""


from src.feature_extraction import mfcc, chroma
import numpy as np
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions


music, labels = load()
mfcc_features = []
chroma_features = []
for m in music:
    mfcc_features.append(mfcc.extract_mfcc(m[0],m[1]))
    chroma_features.append(chroma.extract_chroma(m[0],m[1]))
mfcc_features = reduce_dimensions(mfcc_features, 4)
chroma_features = reduce_dimensions(chroma_features, 3)

features = [np.concatenate((mfcc_features[i], chroma_features[i])) for i in range(len(music))]
print(features)
train_svm(features, labels)
