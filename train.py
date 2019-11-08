from model import get_model
import keras
import numpy as np
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'out_size': 64,
          'bow_size': 100,
          'shuffle': True}

#bow and labels are dictionaries from key to vector


class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'

    def __init__(self, list_IDs, labels, bow, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.bow= bow
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes= None
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

        return X, Xprime, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, params['out_size']))
        Xprime= np.empty((self.batch_size,  params['bow_size']))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.load('data/' + ID + '.npy')
            # Store class
            Xprime[i, ]= self.bow[ID]
            y[i, ] = self.labels[ID]

        return X, Xprime, y


def train():
    model = get_model(params['dim'], params['bow_size'], params['out_size'])
    training_generator = DataGenerator(partition['train'], y, bow, **params)
    model.fit_generator(generator=training_generator,
                        use_multiprocessing=True,
                        workers=6)
# First argument is input shape of spectrogram, second argument is PCA ke baad waala SHAPE, output shape final
