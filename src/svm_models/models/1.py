import src.feature_fusion.Fuser as f
from src.feature_extraction.FeatureExtractor import FeatureExtractor
from src.svm_models.Trainer import Trainer
import logging


trainer = Trainer(logging.getLogger(__name__))
trainer.load()
music, _ = trainer.get_data()
fuser = f.Fuser()
extractor = FeatureExtractor(music)

fuser.load(extractor.get_chroma())
print(fuser._loaded)
print(fuser.run_pca())

#fuser.fuse()

