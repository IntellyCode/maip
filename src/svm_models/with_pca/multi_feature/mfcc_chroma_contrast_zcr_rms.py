"""
A kernel SVM combining mfcc, chroma, bandwidth, zcr and rms to classify music genres

Accuracy with 10/13 features: 82%
Accuracy with 11/26 features: 80% -- Most Efficient

Using Grid Search:
Best Parameters: {'C': 1, 'gamma': 0.001, 'kernel': 'linear'}
Best Score: 0.8541666666666666
Accuracy: 83%
"""

import numpy as np
from src.svm_models.with_pca.multi_feature.util.pca_bar_chart import plot_pca_bar_chart
from src.feature_extraction import rms,spec_rolloff,spec_bandwidth,mfcc,chroma
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions
import logging


# Load music and labels
music, labels = load()

# get logger
logger = logging.getLogger(__name__)

# Extract RMS features
features_mfcc = []
features_chroma = []
features_rms = []
features_rolloff = []
features_bandwidth = []
i = 0
for m in music:
    i += 1
    logger.info(f"Analysing track {i}/300")
    features_mfcc.append(mfcc.extract_mfcc(m[0],m[1]))
    features_chroma.append(chroma.extract_chroma(m[0],m[1]))
    features_rms.append(rms.extract_rms(m[0]))
    features_rolloff.append(spec_rolloff.extract_spectral_rolloff(m[0],m[1]))
    features_bandwidth.append(spec_bandwidth.extract_spectral_bandwidth(m[0],m[1]))

logger.info("Reduce dimensions of MFCC")
features_mfcc = reduce_dimensions(features_mfcc,3)
logger.info("Reducing dimensions of Chroma")
features_chroma = reduce_dimensions(features_chroma,4)
logger.info("Reducing dimensions of RMS")
features_rms = reduce_dimensions(features_rms,1)
logger.info("Reducing dimensions of Rolloff")
features_rolloff = reduce_dimensions(features_rolloff,2)
logger.info("Reducing dimensions of Bandwidth")
features_bandwidth = reduce_dimensions(features_bandwidth,1)
logger.info("Reducing dimensions of Contrast")
features = [np.concatenate((
    features_mfcc[i],
    features_chroma[i],
    features_rms[i],
    features_rolloff[i],
    features_bandwidth[i],
)) for i in range(len(features_rms))
]
"""
logger.info("Plotting PCA Bar Chart")
feature_names = [
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8",
    "chroma_1", "chroma_2", "chroma_3", "chroma_4", "chroma_5", "chroma_6", "chroma_7", "chroma_8",
    "chroma_9", "chroma_10", "chroma_11", "chroma_12",
    "rms_1", "rms_2",
    "rolloff_1", "rolloff_2",
    "bandwidth_1", "bandwidth_2"
]
plot_pca_bar_chart(features, feature_names)
"""

# features_i = reduce_dimensions(features,11)
train_svm(features, labels)
"""
for i in range(1,27):
    logger.info(f"Reducing total dimensions to {i}")
    features_i = reduce_dimensions(features,i)
    train_svm(features_i,labels)
"""
