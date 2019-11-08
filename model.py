from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Dropout
from keras.losses import cosine_proximity
from keras.optimizers import Adam


def get_model(input_shape, output_size):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=4, activation='relu', use_bias=True, input_shape=input_shape))
    model.add(MaxPool1D(pool_size=4, padding='same', data_format='channels_first'))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=512, kernel_size=4, activation='relu', use_bias=True))
    model.add(MaxPool1D(pool_size=4, padding='same', data_format='channels_first'))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=1024, kernel_size=4, activation='relu', use_bias=True))
    model.add(MaxPool1D(pool_size=4, padding='same', data_format='channels_first'))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=1024, kernel_size=4, activation='relu', use_bias=True))

    model.compile(loss=cosine_proximity, optimizer=Adam, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    pass