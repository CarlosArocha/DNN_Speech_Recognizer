# Checking the json file with the original data to adapt it to my storage
# location for the same data, and updating the file in a new file named:
# 'train_corpus_local.json'
from data_generator import vis_train_features, plot_raw_audio
from data_generator import plot_spectrogram_feature
from data_generator import plot_mfcc_feature
from IPython.display import Markdown, display
from mpl_toolkits.axes_grid1 import make_axes_locatable

from keras.backend import set_session
import keras
from keras.optimizers import Adam
import tensorflow as tf

# import NN architectures for speech recognition
from sample_models import *
# import function for training acoustic model
from train_utils import train_model

import matplotlib.pyplot as plt
import json
import os
import numpy as np
import soundfile as sf
import miniaudio
import array


def json_to_local(desc_file='train_corpus.json',
                  new_file='train_corpus_local.json', folder='dev-clean'):
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
data_directory_test = '/Volumes/OutSSD/DATA/NLP/LibriSpeech/dev-clean'
data_directory_valid = '/Volumes/OutSSD/DATA/NLP/LibriSpeech/test-clean'

# FIRST STEP: Change original json file to a file with the local addresses of
# sound files
# json_to_local()
# json_to_local(desc_file='valid_corpus.json',
#               new_file='valid_corpus_local.json')

# SECOND STEP: TO CONVERT THE DATA FROM FLAC TO WAV,we can use sounfile library
# lib='sf' or miniaudion library lib='mini'
# convert_flac_files(data_directory_valid, lib='sf')
# convert_flac_files(data_directory_valid, lib='sf')

###############################################################################

# (todo) json_file = 'train_corpus_local.json'
# extract label and audio features for a single training example
# (todo) vis_text, vis_raw_audio, vis_mfcc_feature, \
# (todo)     vis_spectrogram_feature, vis_audio_path = \
# (todo)     vis_train_features(desc_file=json_file)

###############################################################################


# plot audio signal
# (todo) plot_raw_audio(vis_raw_audio)
# (todo) plt.show(block=False)
# print length of audio signal
# (todo) print('**Shape of Audio Signal** : ' + str(vis_raw_audio.shape))
# print transcript corresponding to audio clip
# (todo) print('**Transcript** : ' + str(vis_text))
# play the audio file
# play_back_file(vis_audio_path)


###############################################################################

# plot normalized spectrogram
# (t) plot_spectrogram_feature(vis_spectrogram_feature)
# print shape of spectrogram
# (t) print('**Shape of Spectrogram** : ' + str(vis_spectrogram_feature.shape))

###############################################################################

# plot normalized MFCC
# (todo) plot_mfcc_feature(vis_mfcc_feature)
# print shape of MFCC
# (todo) print('**Shape of MFCC** : ' + str(vis_mfcc_feature.shape))

###############################################################################
# Creating the model:
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.compat.v1.Session(config=config))

# change to 13 if you would like to use MFCC features
'''model_end = final_model(input_dim=161,
                        filters=5,
                        kernel_size=11,
                        conv_stride=1,
                        conv_border_mode='valid',
                        recur_layers=1,
                        dropout_cnn=0.1,
                        dropout_gru=0.5,
                        use_bias=False,
                        bi=False,
                        units=100)'''

model_end = simple_rnn_model(input_dim=161)

# change spectrogram to False if you would like to use MFCC features
# print(tf.config.list_physical_devices())
# print("Num GPUs Available: ",
#      len(tf.config.experimental.list_physical_devices('GPU')))

train_model(input_to_softmax=model_end,
            pickle_path='model_end.pickle',
            save_model_path='model_end.h5',
            epochs=2,
            optimizer=Adam(learning_rate=0.001,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07,
                           amsgrad=False,
                           name="Adam",),
            spectrogram=True,
            verbose=1)

'''mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, verbose=1)
'''
