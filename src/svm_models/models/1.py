import src.feature_fusion.Fuser as f
from src.feature_extraction.FeatureExtractor import FeatureExtractor
from src.svm_models.Trainer import Trainer
import logging


trainer = Trainer(logging.getLogger(__name__))
trainer.load()
music, _ = trainer.get_data()
fuser = f.Fuser()
extractor = FeatureExtractor(music)

logging.info("Loading Chroma")
fuser.load(extractor.get_chroma())
logging.info("Loading MFCC")
fuser.load(extractor.get_mfcc())
logging.info("Loading Spectral Contrast")
fuser.load(extractor.get_spec_contrast())
logging.info("Loading ZCR")
fuser.load(extractor.get_zcr())
logging.info("Loading Bandwidth")
fuser.load(extractor.get_spec_bandwidth())
logging.info("Loading Kurtosis")
fuser.load(extractor.get_spec_kurtosis())
logging.info("Loading Rolloff")
fuser.load(extractor.get_spec_rolloff())
logging.info("Loading Centroid")
fuser.load(extractor.get_spec_centroid())
logging.info("Loading RMS")
fuser.load(extractor.get_rms())
logging.info("Loading Tonnetz")
fuser.load(extractor.get_tonnetz())
logging.info("Loading Flatness")
fuser.load(extractor.get_flatness())
fuser.fuse()
logging.info("Reducing down to 20")
features = fuser.get_features()
fuser = f.Fuser()
fuser.load(features)
fuser.run_pca(20)
fuser.fuse()

trainer.grid_search(fuser.get_features())
print(trainer)

