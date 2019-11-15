import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from librosa.core import cqt
import sys
import json
from librosa.display import specshow
from slugify import slugify
from librosa import amplitude_to_db

def get_cqt(filename):
    wav, sr = sf.read(filename)
    cqts = cqt(np.asfortranarray(wav), sr=sr, hop_length=1024)[np.newaxis, :, :]
    # array = np.vstack(mels)
    # print(array.shape)
    # print(sr)
    return cqts, sr


if __name__ == '__main__':
    song_id_name = json.load(open('id_to_name.json'))
    song_name_id = json.load(open('name_to_id.json'))
    # song_name_id = {name.replace("_mono.wav.down.wav", "down.wav"): id for name, id in song_name_id.items()}
    # song_id_name = {id: name.replace("_mono.wav.down.wav", "down.wav") for id, name in song_id_name.items()}
    # for x in song_name_id.keys():
    #     print(os.path.isfile(os.path.join('wav', x)))
    # json.dump(song_name_id, open('name_to_id.json', 'w'))
    # json.dump(song_id_name, open('id_to_name.json', 'w'))
    # with open('actualsonglist.txt', 'r') as f:
    #     for line in f:
    #         words = line.split('<SEP>')
    #         song_names.append(slugify(words[-1]))
    #         song_id.append(words[1])
    # print(song_names)
    # print(song_id)
    # exit(0)
    if len(sys.argv) != 3:
        print("Usage: python3 spectrogram.py input_dir output_dir")
        exit(-1)
    datadir = sys.argv[1]
    output_dir = sys.argv[2]
    # # for name in song_name_id.keys():
    # #     x = os.path.isfile(os.path.join(datadir, name))
    # #     if not x:
    # #         print(name)
    # #
    # # exit(0)
    # song_id_to_index = json.load(open('songidToIndex.json'))
    # song_index_to_id = {value: key for key, value in song_id_to_index.items()}
    # for ID, file in song_id_name.items():
    #     print(file)
    #     if not os.path.isfile(os.path.join(datadir, file)):
    #         print("{}           not present".format(file))
    #         continue
    #     new_file_name = str(song_id_to_index[ID]) + '.npy'
    #     if os.path.isfile(os.path.join(output_dir, new_file_name)):
    #         print("skipping                {}".format(file))
    #         continue
    #     array, sr = get_cqt(os.path.join(datadir, file))
    #     print(array.shape)
    #     np.save(os.path.join(output_dir, new_file_name), array, allow_pickle=True)

    for x in os.listdir(output_dir):
        file = os.path.join(output_dir, x)
        y = np.load(file)[0, :, :]
        specshow(amplitude_to_db(np.abs(y), ref=np.max), sr=22050, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrum')
        plt.tight_layout()
        plt.show()
        # if y.shape != (1, 84, 323):
            # print(song_id_name[song_index_to_id[int(x[:-4])]], y.shape)
