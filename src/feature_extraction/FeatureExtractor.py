import librosa.feature as lf
import librosa.beat as lb
import librosa.effects as lef
import numpy as np
import pywt


class FeatureExtractor:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr

    def set_audio(self, y, sr):
        self.y = y
        self.sr = sr

    def chroma(self, n_chroma=12):
        return lf.chroma_stft(y=self.y, sr=self.sr, n_chroma=n_chroma)

    def flatness(self):
        return lf.spectral_flatness(y=self.y)

    def beats(self):
        beats = lb.beat_track(y=self.y, sr=self.sr)[1]
        return np.diff(beats)

    def mel_spectrogram(self, n_mels=128):
        return lf.melspectrogram(y=self.y, sr=self.sr, n_mels=n_mels)

    def mfcc(self, n_mfcc=13):
        return lf.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc)

    def rms(self):
        return lf.rms(y=self.y)

    def spec_bandwidth(self):
        return lf.spectral_bandwidth(y=self.y, sr=self.sr)

    def spec_centroid(self):
        return lf.spectral_centroid(y=self.y, sr=self.sr)

    def spec_contrast(self):
        return lf.spectral_contrast(y=self.y, sr=self.sr)

    def spec_rolloff(self, roll_percent=0.85):
        return lf.spectral_rolloff(y=self.y, sr=self.sr, roll_percent=roll_percent)

    def spec_kurtosis(self):
        kurtosis = lf.spectral_flatness(y=self.y) ** 2
        return kurtosis

    def tempo(self):
        return lb.tempo(y=self.y, sr=self.sr)

    def tonnetz(self):
        return lf.tonnetz(y=self.y, sr=self.sr)

    def wavelet(self, wavelet='db1'):
        coeffs, _ = pywt.cwt(self.y, scales=np.arange(1, 129), wavelet=wavelet, sampling_period=1 / self.sr)
        return coeffs

    def zcr(self):
        return lf.zero_crossing_rate(y=self.y)
