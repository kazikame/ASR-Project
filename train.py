from model import get_model
from tensorflow import keras
import numpy as np
import pickle
import json

# bow and labels are dictionaries from key to vector

with open('tfidf/pca.pickle', 'rb') as f:
    bow = pickle.load(f)

y = np.load('item_factors.npy')
valid = np.loadtxt('valid_indices.txt').astype(int)

np.random.seed(0)

np.random.shuffle(valid)

params1 = {'dim': (128, 216, 1),
          'batch_size': 64,
          'out_size': 1024,
          'bow_size': (bow.shape[1], ),
          'shuffle': True}

partition = {
    'train': valid[:7000],
    'test': valid[7000:]}


class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'

    def __init__(self, list_IDs, labels, bow, batch_size=32, dim=(32, 32, 32), shuffle=True):
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.bow = bow
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Xprime, y = self.__data_generation(list_IDs_temp)

        return [X, Xprime], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size, params1['out_size']))
        Xprime = np.empty((self.batch_size, params1['bow_size'][0]))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            temp = np.moveaxis(np.load('index_npy/' + str(ID) + '.npy'), 0, -1)
            X[i, :, :temp.shape[1], :] = temp
            # Store class
            Xprime[i, ] = self.bow[ID, :]
            y[i, ] = self.labels[ID, :]
        # print("X shape      ", X.shape)
        return X, Xprime, y


def train():
    print("start")
    model = get_model(params1['dim'], params1['bow_size'], params1['out_size'])
    training_generator = DataGenerator(partition['train'], y, bow, batch_size=params1['batch_size'], dim=params1['dim'])
    validation_generator = DataGenerator(partition['test'], y, bow, batch_size=params1['batch_size'], dim=params1['dim'])
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)
# First argument is input shape of spectrogram, second argument is PCA ke baad waala SHAPE, output shape final
