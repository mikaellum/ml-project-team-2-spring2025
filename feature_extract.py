###################################
# The goal of this code is to     #
# Extract Features from the array #
# of the mp3 file for traning the #
# Machine Learning Model          #
###################################

# Import Libraries
import sys
import librosa
import numpy as np

# Define Standard Size (For Hop-Length)
nfft = 2048
t = 25

# Define Class
class mp3FeatExtract:
    def __init__(self, title, fs, Ch0, Ch1, genre):
        self.title = title
        self.fs = fs
        self.Ch0 = Ch0
        self.Ch1 = Ch1
        self.genre = genre
    def Features(self):
        # Down-mix to mono and only take the first 10 seconds to make all features equal length
        mono = (self.Ch0[0:self.fs*t] + self.Ch1[0:self.fs*t] )/2
        # Create a Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y = mono, sr = self.fs)
        mel_spectrogram_dB = librosa.amplitude_to_db(mel_spectrogram, ref = np.max)
        # Zero-Crossing Rate
        zero_crossing_rate = sum(librosa.zero_crossings(y = mono, pad=False))
        # Harmonics and Perceptrual
        harm, perc = librosa.effects.hpss(y = mono)
        # Tempo Beats Per Minute
        tempo, _ = librosa.beat.beat_track(y = mono, sr = self.fs)
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y = mono, sr = self.fs)
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y = mono, sr = self.fs)
        # Mel-Frequency Cepstral Coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y = mono, sr = self.fs)
        # Chroma Frequencies (hop length = 2048 samples)
        chromogram = librosa.feature.chroma_stft(y = mono, sr = self.fs, hop_length = nfft)
        # Get the RMS of the file
        rms = librosa.feature.rms(y = mono).mean()
        return mel_spectrogram_dB, zero_crossing_rate, harm, perc, tempo, spectral_centroids, spectral_rolloff, mfccs, chromogram, rms
    def mapGenre(self):
        # Map the genre label to a number from 1 to 13
        match self.genre:
            case 'Pop/Synth-Pop/Indie-Pop':
                return 0
            case 'Rock/Alternative/Punk':
                return 1
            case 'Hip-Hop/Rap':
                return 2
            case 'Folk/Acoustic/Singer-Songwriter':
                return 3
            case 'Ambient/Experimental/Noise':
                return 4
            case 'World/International/Latin':
                return 5
            case 'Metal':
                return 6
            case 'Electronic/Dance':
                return 7
            case 'Other/Undefined':
                return 8
            case 'Reggae/Dub':
                return 9
            case 'Classical/Cinematic/Orchestral':
                return 10
            case 'Jazz/Blues/Soul':
                return 11
            case 'Children/Holiday/Novelty':
                return 12      


        
# Call the class from command line
if __name__ == "__main__":
    if len(sys.argv) == 5:
        title = sys.argv[1]
        fs = sys.argv[2]
        Ch0 = sys.argv[3]
        Ch1 = sys.argv[4]
    obj = mp3FeatExtract(title, fs, Ch0, Ch1)
    mel_spectrogram_dB, zero_crossing_rate, harm, perc, tempo, spectral_centroids, spectral_rolloff, mfccs, chromogram, rms = obj.Features()