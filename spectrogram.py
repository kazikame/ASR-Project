import os
import numpy as np
import matplotlib.pyplot as plt
from librosa import power_to_db
import soundfile as sf
from librosa.feature import melspectrogram
import sys
import json
from librosa.display import specshow
from slugify import slugify

def get_melspectogram(filename):
    wav, sr = sf.read(filename, always_2d=True)
    mels = [melspectrogram(np.asfortranarray(wav[:, i]), sr=sr,
                           hop_length=1024)[np.newaxis, :, :] for i in range(wav.shape[1])]
    array = np.vstack(mels)
    # print(array.shape)
    # print(sr)
    return array, sr


if __name__ == '__main__':
    song_id_name = json.load(open('id_to_name.json'))
    song_name_id = json.load(open('name_to_id.json'))
    # # with open('actualsonglist.txt', 'r') as f:
    # #     for line in f:
    # #         words = line.split('<SEP>')
    # #         song_names.append(slugify(words[-1]))
    # #         song_id.append(words[1])
    # # print(song_names)
    # # print(song_id)
    # # exit(0)
    # if len(sys.argv) != 3:
    #     print("Usage: python3 spectrogram.py input_dir output_dir")
    #     exit(-1)
    datadir = sys.argv[1]
    output_dir = sys.argv[2]
    # # for name in song_name_id.keys():
    # #     x = os.path.isfile(os.path.join(datadir, name))
    # #     if not x:
    # #         print(name)
    # #
    # # exit(0)
    song_id_to_index = json.load(open('songidToIndex.json'))
    song_index_to_id = {value: key for key, value in song_id_to_index.items()}
    # for ID, file in song_id_name.items():
    #     print(file)
    #     if not os.path.isfile(os.path.join(datadir, file)):
    #         print("{}           not present".format(file))
    #         continue
    #     new_file_name = str(song_id_to_index[ID]) + '.npy'
    #     if os.path.isfile(os.path.join(output_dir, new_file_name)):
    #         print("skipping                {}".format(file))
    #         continue
    #     array, sr = get_melspectogram(os.path.join(datadir, file))
    #     np.save(os.path.join(output_dir, new_file_name), array, allow_pickle=True)
    for x in os.listdir(output_dir):
        file = os.path.join(output_dir, x)
        y = np.load(file)
        if y.shape != (1, 128, 216):
            print(song_id_name[song_index_to_id[int(x[:-4])]], y.shape)

# x = os.listdir(audio_dir)[0]
# x = "this_time_i_know_preview.wav"
# mel, sr = get_melspectogram(x)
# for i in range(mel.shape[0]):
#     S_dB = power_to_db(mel[i, :, :], ref=np.max)
#     specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
#     plt.tight_layout()
#     plt.show()
#     plt.close()
