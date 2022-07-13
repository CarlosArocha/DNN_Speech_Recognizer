# Checking the json file with the original data to adapt it to my storage location
# for the same data, and updating the file in a new file named: 'train_corpus_local.json'
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
            keys.append(spec['key'].replace('/data/nlpnd_projects/', new_base_path))
            durations.append(spec['duration'])
            labels.append(spec['text'])
        json_line_file.close()
    with open(new_file, 'w') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                              'text': labels[i]})
            out_file.write(line + '\n')
        out_file.close()

    print('New file ready')

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
    result.num_frames = int(len(converted_frames) / \
                                 result.nchannels / \
                                 result.sample_width)
    result.samples.frombytes(converted_frames)
    miniaudio.wav_write_file(new_file, result)
    # In case you want to print the output file info: decomment
    # output_info = miniaudio.get_file_info(new_file)
    # print(output_info)


def convert_flac_files(data_directory):

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
                            flac_to_wav_wminiaudio(filename,
                                        os.path.join(speaker_folder, new_file))
                # When sounfile library is fixed use:
                # data, samplerate = sf.read(os.path.join(speaker_folder, file))
                # sf.write(new_file, data, samplerate, 'PCM_16')

    print('.flac files converted to .wav')

wk_directory='/Users/carlos/Dropbox/AI/GITHUB/UDACITY/NLP/DNN_Speech_Recognizer'
os.chdir(wk_directory)


directory = '/Volumes/OutSSD/DATA/NLP/LibriSpeech/dev-clean'
convert_flac_files(directory, )
