import os
import numpy as np
import matplotlib.pyplot as plt
from librosa import power_to_db
import soundfile as sf
from librosa.feature import melspectrogram
import sys
from librosa.display import specshow
from x import sluggify

def get_melspectogram(filename):
    wav, sr = sf.read(filename, always_2d=True)
    mels = [melspectrogram(np.asfortranarray(wav[:, i]), sr=sr,
                           hop_length=1024)[np.newaxis, :, :] for i in range(wav.shape[1])]
    array = np.vstack(mels)
    # print(array.shape)
    # print(sr)
    return array, sr


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python3 spectrogram.py input_dir output_dir")
        exit(-1)
    datadir = sys.argv[1]
    output_dir = sys.argv[2]

    for file in os.listdir(datadir):
        print(file)
        if os.path.isfile(os.path.join(output_dir, file[:-3] + 'npy')):
            continue
        array, sr = get_melspectogram(os.path.join(datadir, file))
        np.save(os.path.join(output_dir, file[:-3] + 'npy'), array, allow_pickle=True)

# x = os.listdir(audio_dir)[0]
# x = "this_time_i_know_preview.wav"
# mel, sr = get_melspectogram(x)
# for i in range(mel.shape[0]):
#     S_dB = power_to_db(mel[i, :, :], ref=np.max)
#     specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
#     plt.tight_layout()
#     plt.show()
#     plt.close()
