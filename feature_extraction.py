import librosa
import sklearn
import csv
import os
import numpy as np
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
import librosa.display


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

    if show:
        # display Spectrogram
        X = librosa.stft(signal)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=smpl_rate, x_axis='time', y_axis='hz')
        plt.colorbar()

        # display spectral centroids
        # Computing the time variable for visualization
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)

        # Normalising the spectral centroid for visualisation
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Plotting the Spectral Centroid along the waveform
        librosa.display.waveplot(signal, sr=smpl_rate, alpha=0.4)
        plt.plot(t, normalize(spectral_centroids), color='r')

        # Plotting specrtal rolloff
        plt.plot(t, normalize(spectral_rolloff), color='r')

        # Plotting MFCC
        librosa.display.specshow(mfccs, sr=smpl_rate, x_axis='time')

        # Plotting chroma_stft
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')

        plt.show()



    return zero_crossing_rate, spectral_centroids, spectral_rolloff, mfccs, chromagram, temp, beat


def kalman_tempo_beat(filename):
    # Load the file
    signal, smpl_rate = librosa.load(filename)

    # Extract tempo and beats estimation
    tempo, beats = librosa.beat.beat_track(y=signal, sr=smpl_rate)
    return tempo, beats


if __name__ == "__main__":
    path = 'dataset_clips'
    show = False     # Set to true for graphs

    # Generate header for .csv file
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
            for filename in tqdm(os.listdir(os.path.join(path + "/" + g + '_New'))):

                # Full path to file
                filename_ = os.path.join(path + "/" + g + '_New/' + filename)

                # Remove gaps in filename for ease of separation while writing to .csv
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
                    writer = csv.writer(file_)
                    writer.writerow(feat.split())
                    file_.close()  # Important to close file before reopening




