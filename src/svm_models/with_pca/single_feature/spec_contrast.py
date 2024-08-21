"""
Extract spectral contrast features and train SVMs with multiple pca components

-- This uses mean, std, min, max
Accuracy with 1 feature: 67%
Accuracy with 4 features: 72%
Accuracy with 7 features: 78% -- Most Efficient
Accuracy with 10 features: 83% -- Best Performance
Accuracy with 13 features: 82%
Accuracy with 16 features: 82%

-- This uses mean, std
Accuracy with 1 feature: 63%
Accuracy with 4 features: 66%
Accuracy with 7 features: 70%
Accuracy with 10 features: 72%
"""


from src.feature_extraction.spec_contrast import extract_spectral_contrast
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions
import logging


music, labels = load()
features = []
for m in music:
    # each feature is an array of 28 elements (mean, std, min, max) * 7
    features.append(extract_spectral_contrast(m[0], m[1]))

for i in range(1, 29, 3):
    features_i = reduce_dimensions(features, i)
    logger = logging.getLogger(__name__)
    logger.info(f"Reduce dimensions down to {i} features")
    train_svm(features_i, labels)
