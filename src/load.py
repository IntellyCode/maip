import logging
import os
import librosa.core as lc


def load(get_features):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s -  %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    genres = ["classical", "reggae", "rock"]
    features = []
    labels = []
    for genre in genres:
        for file in os.listdir("/Users/zeniosd/Documents/Programs/Python/maip/data/raw/" + genre):
            path = os.path.join("/Users/zeniosd/Documents/Programs/Python/maip/data/raw/" + genre, file)
            logger.info(f"Loading {genre} features for {path}")
            y, sr = lc.load(path)
            feature = get_features(y, sr)
            features.append(feature)
            labels.append(genre)

    return features, labels
