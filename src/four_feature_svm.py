"""
Uses an SVM with RFB kernel to distinguish between classical, reggae and rock genres.

Model has a 77% accuracy.
"""


import librosa.feature as lf
import numpy as np
import os
import librosa.core as lc
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime


def get_mfcc_feature(y: np.ndarray, sr: int):
    """

    :param y: audio time series
    :param sr: audio sampling rate of y
    :return mfcc feature: Numpy Array
    """
    mfcc = np.array(lf.mfcc(y=y, sr=sr))
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_feature = np.concatenate((mfcc_mean, mfcc_min, mfcc_max))

    return mfcc_feature


def get_melspectrogram(y: np.ndarray, sr: int):
    """
    :param y: audio time series
    :param sr: audio sampling rate of y
    :return melspectrogram: Numpy Array
    """
    melspectrogram = np.array(lf.melspectrogram(y=y, sr=sr))
    melspectrogram_mean = melspectrogram.mean(axis=1)
    melspectrogram_min = melspectrogram.min(axis=1)
    melspectrogram_max = melspectrogram.max(axis=1)
    melspectrogram_feature = np.concatenate((melspectrogram_mean, melspectrogram_min, melspectrogram_max))

    return melspectrogram_feature


def get_chroma(y: np.ndarray, sr: int):
    """
    :param y: audio time series
    :param sr: audio sampling rate of y
    :return chroma: Numpy Array
    """
    chroma = np.array(lf.chroma_stft(y=y, sr=sr))
    chroma_mean = chroma.mean(axis=1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)
    chroma_feature = np.concatenate((chroma_mean, chroma_min, chroma_max))

    return chroma_feature


def get_tonnetz(y: np.ndarray, sr: int):
    """
    :param y: audio time series
    :param sr: audio sampling rate of y
    :return tonnetz: Numpy Array
    """

    tntz = np.array(lf.tonnetz(y=y, sr=sr))
    tntz_mean = tntz.mean(axis=1)
    tntz_min = tntz.min(axis=1)
    tntz_max = tntz.max(axis=1)
    tntz_feature = np.concatenate((tntz_mean, tntz_min, tntz_max))

    return tntz_feature


def get_features(audio: np.ndarray, sr: int):
    mfcc = get_mfcc_feature(audio, sr)
    melspec = get_melspectrogram(audio, sr)
    chroma = get_chroma(audio, sr)
    tonnetz = get_tonnetz(audio, sr)

    feature = np.concatenate((mfcc, melspec, chroma, tonnetz))
    return feature


now = datetime.now()
logger = logging.getLogger(__name__)
path = f"../results/logs/four_feature_svm-{now}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s -  %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(path),
    ]

)
genres = ["classical","reggae","rock"]
features = []
labels = []
for genre in genres:
    for file in os.listdir("../data/raw/" + genre):
        path = os.path.join("../data/raw/" + genre, file)
        logger.info(f"Loading {genre} features for {path}")
        y, sr = lc.load(path)
        feature = get_features(y, sr)
        features.append(feature)
        labels.append(genre)


X = np.array(features)
le = LabelEncoder()
y = le.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel="rbf")
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logger.info(f"Classification report: {classification_report(y_test, y_pred, target_names=le.classes_)}")




