# Checking the json file with the original data to adapt it to my storage location
# for the same data, and updating the file in a new file named: 'train_corpus_local.json'
import json
import os
import numpy as np
import soundfile as sf

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

def flac_to_wav(data_directory):
    '''wav_file_count = 0
    save_path = os.path.join(data_directory, group, speaker, chapter)
    idtag = speaker+'-'+chapter
    transcript_filename = idtag + '.trans.txt'
    os.makedirs(save_path, exist_ok=True)
    outfile = open(os.path.join(save_path, transcript_filename), 'w')
    save_path = os.path.join(data_directory, group, speaker, chapter)
    for file in os.listdir(input_directory):
        if file.endswith(".wav"):
            data, samplerate = sf.read(os.path.join(input_directory,file))
            sf.write('testwavout.wav',data,samplerate)
            # save the file to its new place
            ident = idtag + '-' + '{:04d}'.format(wav_file_count)
            new_filename = ident+'.wav'
            print(ident)
            os.replace('testwavout.wav',os.path.join(save_path,new_filename))
            wav_file_count += 1
            outfile.write(ident+' \n')
    outfile.close()'''

    for group in os.listdir(data_directory):
        if not group.startswith('.'):
            group_folder = os.path.join(data_directory, group)
            for speaker in os.listdir(group_folder):
                if not speaker.startswith('.'):
                    speaker_folder = os.path.join(data_directory, group, speaker)
                    for file in os.listdir(speaker_folder):
                        if (not file.startswith('.')) and file.endswith('.flac'):
                            filename = os.path.join(speaker_folder, file)
                            print(filename)
                            data, samplerate = sf.read(os.path.join(speaker_folder, file))
                            sf.write(file[:-4]+'wav', data, samplerate, 'PCM_16')

    print('.flac files converted to .wav')

wk_directory='/Users/carlos/Dropbox/AI/GITHUB/UDACITY/NLP/DNN_Speech_Recognizer'
os.chdir(wk_directory)

# json_to_local()

print(sf.available_formats())
print(sf.available_subtypes('FLAC'))
print(sf._libname)

directory = '/Volumes/OutSSD/DATA/NLP/LibriSpeech/dev-clean'
flac_to_wav(directory, )
