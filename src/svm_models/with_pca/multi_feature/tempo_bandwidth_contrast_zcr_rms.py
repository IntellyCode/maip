"""
A kernel SVM combining tempo, bandwidth, contrast, zcr and rms to classify music genres



Accuracy with 4 features: 80%
Accuracy with 5 features: 83%
Accuracy with 6 features: 87%
Accuracy with 11 features: 87%

Accuracy of 83% with parameter tuning C:1, gamma:1, kernel:poly
"""

import numpy as np
from src.svm_models.with_pca.multi_feature.util.pca_bar_chart import plot_pca_bar_chart
from src.feature_extraction import rms,spec_rolloff,spec_bandwidth,spec_contrast,tempo
from src.feature_extraction.util import scale
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions
import logging


# Load music and labels
music, labels = load()

#get logger
logger = logging.getLogger(__name__)

# Extract RMS features
features_rms = []
features_rolloff = []
features_bandwidth = []
features_contrast = []
features_tempo = []
i = 0
for m in music:
    i += 1
    logger.info(f"Analysing track {i}/300")
    features_rms.append(rms.extract_rms(m[0]))
    features_rolloff.append(spec_rolloff.extract_spectral_rolloff(m[0],m[1]))
    features_bandwidth.append(spec_bandwidth.extract_spectral_bandwidth(m[0],m[1]))
    features_contrast.append(spec_contrast.extract_spectral_contrast(m[0],m[1]))
    features_tempo.append(tempo.extract_tempo(m[0],m[1]))

logger.info("Reducing dimensions of RMS")
features_rms = reduce_dimensions(features_rms,2)
logger.info("Reducing dimensions of Rolloff")
features_rolloff = reduce_dimensions(features_rolloff,2)
logger.info("Reducing dimensions of Bandwidth")
features_bandwidth = reduce_dimensions(features_bandwidth,2)
logger.info("Reducing dimensions of Contrast")
features_contrast = reduce_dimensions(features_contrast,4)
features_tempo = np.array(scale(np.array(features_tempo))).reshape(-1,1)
features = [np.concatenate((
    features_rms[i],
    features_rolloff[i],
    features_bandwidth[i],
    features_contrast[i],
    features_tempo[i])) for i in range(len(features_rms))
]
logger.info("Plotting PCA Bar Chart")

features_i = reduce_dimensions(features, 6)
train_svm(features_i, labels)

"""
# Feature names corresponding to each column in X
feature_names = [
        'rms_1', 'rms_2',
        'rolloff_1', 'rolloff_2',
        'bandwidth_1', 'bandwidth_2',
        'contrast_1', 'contrast_2', 'contrast_3', 'contrast_4',
        'tempo_1'
    ]
plot_pca_bar_chart(features,feature_names)
for i in range(1,12):
    logger.info(f"Reducing total dimensions to {i}")
    features_i = reduce_dimensions(features,i)
    train_svm(features_i,labels)
"""