import numpy as np

from src.feature_extraction.tempo import extract_tempo
from src.feature_extraction.zcr import extract_zero_crossing_rate
from src.load import load
import logging
from src.feature_extraction.util import scale
# Load music and labels
music, labels = load()

# get logger
logger = logging.getLogger(__name__)

# Extract RMS features
tempo = []
zcr = []
i = 0
for m in music:
    i += 1
    logger.info(f"Analysing track {i}/300")
    # each feature is an array of 4 elements (mean, std, min, max)
    #zcr.append(extract_zero_crossing_rate(m[0]))
    tempo.append(extract_tempo(m[0],m[1]))

print(zcr)
tempo = scale(np.array(tempo))
tempo =np.array(tempo).reshape(-1, 1)
print(tempo)
