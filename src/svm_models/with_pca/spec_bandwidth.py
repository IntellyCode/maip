"""
Extract Spectral Bandwidth features and train SVMs with multiple PCA components

Accuracy with 2 features: 58%

"""

from src.feature_extraction.spec_bandwidth import extract_spectral_bandwidth
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions
import logging


# Load music and labels
music, labels = load()

# Extract Spectral Bandwidth features
features = []
for m in music:
    # each feature is an array of 4 elements (mean, std, min, max)
    features.append(extract_spectral_bandwidth(m[0], sr=m[1]))

# Train SVMs with different PCA components
for i in range(1, 5):
    features_i = reduce_dimensions(features, i)
    logger = logging.getLogger(__name__)
    logger.info(f"Reduce dimensions down to {i} features")
    train_svm(features_i, labels)
