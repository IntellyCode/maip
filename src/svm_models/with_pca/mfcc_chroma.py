"""
Kernel SVM which uses 4 components of the mfcc and 3 components of the chrome to classify music genres

Accuracy:
"""


from src.feature_extraction import mfcc, chroma
import numpy as np
from src.load import load
from src.svm_models.train_svm import train_svm


def extract_features(y, sr):
    mfcc_features = mfcc.extract_mfcc_rd(y, sr)
    chroma_features = chroma.extract_chroma_rd(y, sr)
    return np.concatenate((mfcc_features,chroma_features))


features, labels = load(extract_features)
train_svm(features, labels)