import os
import numpy as np
import matplotlib.pyplot as plt
from librosa import power_to_db
import soundfile as sf
from librosa.feature import melspectrogram
from librosa.display import specshow


audio_dir = "/Users/yashs20/iitb.ac.in/audio_wav"


def get_melspectogram(filename):
    wav, sr = sf.read(os.path.join(audio_dir, filename))
    mels = [melspectrogram(np.asfortranarray(wav[:, i]), sr=sr, fmax=8000)[np.newaxis, :, :] for i in range(wav.shape[1])]
    return np.vstack(mels), sr


# x = os.listdir(audio_dir)[0]
x = "this_time_i_know_preview.wav"
mel, sr = get_melspectogram(x)
for i in range(mel.shape[0]):
    S_dB = power_to_db(mel[i, :, :], ref=np.max)
    specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.tight_layout()
    plt.show()
    plt.close()
