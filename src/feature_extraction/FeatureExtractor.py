import librosa.feature as lf
import librosa.beat as lb
import librosa.effects as lef
import numpy as np
import pywt


class FeatureExtractor:
    class _Music:
        def __init__(self, y, sr):
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

    def __init__(self, music):
        self._music = [FeatureExtractor._Music(y, sr) for (y, sr) in music]

    def get_record(self, i):
        return self._music[i]

    def get_chroma(self):
        arr = [m.chroma() for m in self._music]
        return arr

    def get_flatness(self):
        arr = [m.flatness() for m in self._music]
        return arr

    def get_beats(self):
        arr = [m.beats() for m in self._music]
        return arr

    def get_mel_spectrogram(self):
        arr =[m.mel_spectrogram() for m in self._music]
        return arr

    def get_mfcc(self):
        arr = [m.mfcc() for m in self._music]
        return arr

    def get_rms(self):
        arr = [m.rms() for m in self._music]
        return arr

    def get_spec_bandwidth(self):
        arr = [m.spec_bandwidth() for m in self._music]
        return arr

    def get_spec_centroid(self):
        arr = [m.spec_centroid() for m in self._music]
        return arr

    def get_spec_contrast(self):
        arr = [m.spec_contrast() for m in self._music]
        return arr

    def get_spec_rolloff(self):
        arr = [m.spec_rolloff() for m in self._music]
        return arr

    def get_spec_kurtosis(self):
        arr = [m.spec_kurtosis() for m in self._music]
        return arr

    def get_tempo(self):
        arr = [m.tempo() for m in self._music]
        return arr

    def get_tonnetz(self):
        arr = [m.tonnetz() for m in self._music]
        return arr

    def get_wavelet(self):
        arr=[m.wavelet() for m in self._music]
        return arr

    def get_zcr(self):
        arr = [m.zcr() for m in self._music]
        return arr

