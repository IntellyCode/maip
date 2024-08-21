"""
Extract spectral rolloff features and train SVMs with multiple pca components

Accuracy with 1 feature: 65% -- Best Performance

"""


from src.feature_extraction.zcr import extract_zero_crossing_rate
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions
import logging


music, labels = load()
features = []
for m in music:
    # each feature is an array of 4 elements (mean, std, min, max)
    features.append(extract_zero_crossing_rate(m[0]))

for i in range(1, 5):
    features_i = reduce_dimensions(features, i)
    logger = logging.getLogger(__name__)
    logger.info(f"Reduce dimensions down to {i} features")
    train_svm(features_i, labels)

