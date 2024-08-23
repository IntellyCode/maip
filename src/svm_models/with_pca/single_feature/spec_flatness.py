"""
Accuracy: 25%
"""

from src.feature_extraction.spec_flatness import extract_spectral_flatness
from src.load import load
from src.svm_models.train_svm import train_svm
from src.feature_extraction.util import reduce_dimensions
import logging


music, labels = load()
features = []
logger = logging.getLogger(__name__)
for m in music:
    logger.info(f"Appending music feature of {m}")
    # 4 feature vector (mean, std, min,max)
    features.append(extract_spectral_flatness(m[0], m[1]))

train_svm(features, labels)
