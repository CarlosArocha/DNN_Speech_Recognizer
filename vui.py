# Checking the json file with the original data to adapt it to my storage
# location for the same data, and updating the file in a new file named:
# 'train_corpus_local.json'
from data_generator import vis_train_features, plot_raw_audio
from data_generator import plot_spectrogram_feature
from data_generator import plot_mfcc_feature
from IPython.display import Markdown, display
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import soundfile as sf
import miniaudio
import array


def json_to_local(desc_file='train_corpus.json',
                  new_file='train_corpus_local.json'):
    new_base_path = '/Volumes/OutSSD/DATA/NLP/'
    keys, durations, labels = [], [], []
    with open(desc_file) as json_line_file:
        for json_line in json_line_file:
            spec = json.loads(json_line)
            keys.append(spec['key'].replace('/data/nlpnd_projects/',
                        new_base_path))
            durations.append(spec['duration'])
            labels.append(spec['text'])
        json_line_file.close()
    with open(new_file, 'w') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                              'text': labels[i]})
            out_file.write(line + '\n')
        out_file.close()

    print('New json file ready')


def play_back_file(filename):
    stream = miniaudio.stream_file(filename)
    device = miniaudio.PlaybackDevice()
    device.start(stream)
    input("Audio file playing in the background. Enter to stop playback: ")
    device.close()


def flac_to_wav_wminiaudio(filename, new_file):

    src = miniaudio.decode_file(filename, dither=miniaudio.DitherMode.TRIANGLE)
    # In case you want to print the source file info: decomment
    # print("Source: ", src)
    result = miniaudio.DecodedSoundFile("result", src.nchannels,
                                        src.sample_rate, src.sample_format,
                                        array.array('b'))
    converted_frames = miniaudio.convert_frames(src.sample_format,
                                                src.nchannels, src.sample_rate,
                                                src.samples.tobytes(),
                                                result.sample_format,
                                                result.nchannels,
                                                result.sample_rate)
    # note: currently it is not possible to provide a dithermode
    # to convert_frames()
    result.num_frames = int(len(converted_frames) /
                            result.nchannels /
                            result.sample_width)
    result.samples.frombytes(converted_frames)
    miniaudio.wav_write_file(new_file, result)
    # In case you want to print the output file info: decomment
    # output_info = miniaudio.get_file_info(new_file)
    # print(output_info)


def convert_flac_files(data_directory, lib='mini'):

    for group in os.listdir(data_directory):

        if not group.startswith('.'):
            group_folder = os.path.join(data_directory, group)
            for speaker in os.listdir(group_folder):
                if not speaker.startswith('.'):
                    speaker_folder = os.path.join(data_directory, group,
                                                  speaker)
                    for file in os.listdir(speaker_folder):
                        if (not file.startswith('.')) and \
                                file.endswith('.flac'):
                            filename = os.path.join(speaker_folder, file)
                            new_file = file[:-4]+'wav'
                            new_path = os.path.join(speaker_folder, new_file)
                            if lib == 'mini':
                                flac_to_wav_wminiaudio(filename, new_path)
                            elif lib == 'sf':
                                data, samplerate = sf.read(filename)
                                sf.write(new_path, data, samplerate, 'PCM_16')

    print('.flac files converted to .wav with {}'.format(lib))


wk_directory = '/Users/carlosarocha/Dropbox/AI/GITHUB/UDACITY/NLP/' +\
    'DNN_Speech_Recognizer'
os.chdir(wk_directory)
data_directory = '/Volumes/OutSSD/DATA/NLP/LibriSpeech/dev-clean'

# FIRST STEP: Change original json file to a file with the local addresses of
# sound files
# json_to_local()

# SECOND STEP: TO CONVERT THE DATA FROM FLAC TO WAV,we can use sounfile library
# lib='sf' or miniaudion library lib='mini'
# convert_flac_files(data_directory, lib='sf')

###############################################################################

json_file = 'train_corpus_local.json'
# extract label and audio features for a single training example
vis_text, vis_raw_audio, vis_mfcc_feature, \
    vis_spectrogram_feature, vis_audio_path = \
    vis_train_features(desc_file=json_file)

###############################################################################


# plot audio signal
plot_raw_audio(vis_raw_audio)
plt.show(block=False)
# print length of audio signal
print('**Shape of Audio Signal** : ' + str(vis_raw_audio.shape))
# print transcript corresponding to audio clip
print('**Transcript** : ' + str(vis_text))
# play the audio file
# play_back_file(vis_audio_path)


###############################################################################

# plot normalized spectrogram
plot_spectrogram_feature(vis_spectrogram_feature)
# print shape of spectrogram
print('**Shape of Spectrogram** : ' + str(vis_spectrogram_feature.shape))

###############################################################################

# plot normalized MFCC
plot_mfcc_feature(vis_mfcc_feature)
# print shape of MFCC
print('**Shape of MFCC** : ' + str(vis_mfcc_feature.shape))
