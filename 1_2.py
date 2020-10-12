######################
# 1.2 Fearure Extraction
######################


import librosa
import sklearn


def extract_feat(filename):
    signal, smpl_rate = librosa.load(filename)
    zero_crossings = librosa.zero_crossings(signal, pad=False)
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=smpl_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=smpl_rate)[0]
    mfccs = librosa.feature.mfcc(signal, sr=smpl_rate)
    print(mfccs.shape)



if __name__ == "__main__":
    extract_feat('dataset_clips/Dark_Forest_New/03 - Dohm & Schizoid Bears - Modulation Manipulation.wav_chunk0.wav')