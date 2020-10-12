######################
# 1.2 Feature Extraction
######################


import librosa
import sklearn
import csv
import os
import numpy as np
import sys


def extract_feat(filename):
    # Function to extract features from an audio clip

    # Load the file
    signal, smpl_rate = librosa.load(filename)

    #Extract the features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal, pad=False)
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=smpl_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=smpl_rate)[0]
    mfccs = librosa.feature.mfcc(signal, sr=smpl_rate)
    chromagram = librosa.feature.chroma_stft(signal, sr=smpl_rate, hop_length=512)  #TODO: Check hop_length
    temp, beat = kalman_tempo_beat(filename)
    return zero_crossing_rate, spectral_centroids, spectral_rolloff, mfccs, chromagram, temp, beat


def kalman_tempo_beat(filename):
    # Load the file
    signal, smpl_rate = librosa.load(filename)

    # Extract tempo and beats estimation
    tempo, beats = librosa.beat.beat_track(y=signal, sr=smpl_rate)
    return tempo, beats

if __name__ == "__main__":
    # extract_feat('dataset_clips/Dark_Forest_New/03 - Dohm & Schizoid Bears - Modulation Manipulation.wav_chunk0.wav')
    path = 'dataset_clips'
    #Generate header for .csv file
    header = 'filename  zero_crossing_rate spectral_centroid sprectral_rolloff chroma_stft tempo beat'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    # Open .csv file to store features
    with open('features.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        genres = 'Dark_Forest Full-on Goa Hi_Tech'.split()

        for g in genres:
            for filename in os.listdir(os.path.join(path + "/" + g + '_New')):

                filename_ = os.path.join(path + "/" + g + '_New/' + filename)

                # Remove gaps for ease of separation while writing to .csv
                file_nogaps = filename.replace(" ", "")

                # Extract the features
                zcr, sp_cent, sp_roll, mfcc, chrom, tmpo, bts = extract_feat(filename_)

                # Combine the mean of all the features ready to be written to .csv file
                feat = f'{file_nogaps} {np.mean(zcr)} {np.mean(sp_cent)} {np.mean(sp_roll)} ' \
                          f'{np.mean(chrom)} {np.mean(tmpo)} {np.mean(bts)}'

                # Add mean of mfcc for each dimension
                for m in mfcc:
                    feat += f' {np.mean(m)}'

                # Add genre label
                feat += f' {g}'

                file.close()  # Important to close the file in write mode
                # before opening in read mode

                with open('features.csv', 'a', newline='') as file_:
                    writer = csv.writer(file_, quoting = csv.QUOTE_NONE)
                    writer.writerow(feat.split())
                    file_.close()  # Improtant to close file before reopening




