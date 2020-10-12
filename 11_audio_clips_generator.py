#################
# DSP Project
# Ashima & Nisha
#################

############################
# 1.1 - Clips the audio files
############################


import os
import sys

# Audio clipping for 30 sec
from pydub import AudioSegment
from pydub.utils import make_chunks


def clipAudio(folderNam):
    # Function to clip all audio files present in "folderName" and save it in path/to/folder/"folderNam_clips"

    genreNam = os.path.basename(folderNam)  # Extract genre name

    # 30s clips will be stored in dst_new (e.g: dataset_clips/Dark-Forest)
    dst_new = os.path.join(os.path.dirname(folderNam) + '_clips/' + genreNam + "_New")

    # create the new folder directory if it is not already created
    if not os.path.exists(dst_new):
        os.makedirs(dst_new)

    # Iterate through all the files in the folder
    for dirpath, dirnames, files in os.walk(folderNam):
        print(f'Found directory: {dirpath}')

        for file_name in files:
            music_file = os.path.join(dirpath, file_name)   # path + filename
            myaudio = AudioSegment.from_file(music_file, "wav")

            # Clip the audio file
            chunk_length_ms = 30000  # pydub calculates in millisec
            chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of 30 sec

            # Iterate through all the chunks of each file
            for i, chunk in enumerate(chunks):
                chunk_name = "chunk{0}.wav".format(i)

                # remove .wav and add _chunk{0}
                chunk_file_name = os.path.join(file_name[:-4] + '_' + chunk_name)
                print(chunk_file_name)

                # Export each chunk as .wav file
                chunk.export(os.path.join(dst_new, chunk_file_name), format="wav")


if __name__ == "__main__":

    clipAudio('dataset/Dark_Forest')
    clipAudio('dataset/Full-On')
    clipAudio('dataset/Goa')
    clipAudio('dataset/Hi_Tech')