from keras.models import Sequential, Model
from keras.layers import Dropout, MaxPooling2D, Convolution2D, Input, Lambda, concatenate, Flatten, Dense
import logging
from keras import backend as K
from keras.losses import cosine_proximity
from keras.optimizers import Adam

params = {
        'dropout_factor': 0.5,
        'n_dense': 0,
        'n_dense_2': 0,
        'n_filters_1': 1024,
        'n_filters_2': 1024,
        'n_filters_3': 2048,
        'n_filters_4': 2048,
        'n_filters_5': 0,
        'n_kernel_1': (4, 96),
        'n_kernel_2': (4, 1),
        'n_kernel_3': (4, 1),
        'n_kernel_4': (1, 1),
        'n_kernel_5': (1, 1),
        'n_out': '',
        'n_pool_1': (4, 1),
        'n_pool_2': (4, 1),
        'n_pool_3': (1, 1),
        'n_pool_4': (1, 1),
        'n_pool_5': (1, 1),
        'n_frames': 322,
        'n_mel': 96,
        'architecture': 2,
        'batch_norm': False,
        'dropout': True
}


def get_model(input_shape, output_size):
    inputs = Input(shape=input_shape)

    conv1 = Convolution2D(params["n_filters_1"], params["n_kernel_1"][0],
                          params["n_kernel_1"][1],
                          border_mode='valid',
                          activation='relu',
                          input_shape=(1, params["n_frames"],
                                       params["n_mel"]),
                          init="uniform")
    x = conv1(inputs)
    # logging.debug("Input CNN: %s" % str(inputs.output_shape))
    logging.debug("Output Conv2D: %s" % str(conv1.output_shape))

    pool1 = MaxPooling2D(pool_size=(params["n_pool_1"][0],
                                    params["n_pool_1"][1]))
    x = pool1(x)
    logging.debug("Output MaxPool2D: %s" % str(pool1.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    conv2 = Convolution2D(params["n_filters_2"], params["n_kernel_2"][0],
                          params["n_kernel_2"][1],
                          border_mode='valid',
                          activation='relu',
                          init="uniform")
    x = conv2(x)
    logging.debug("Output Conv2D: %s" % str(conv2.output_shape))

    pool2 = MaxPooling2D(pool_size=(params["n_pool_2"][0],
                                    params["n_pool_2"][1]))
    x = pool2(x)
    logging.debug("Output MaxPool2D: %s" % str(pool2.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    # model.add(Permute((3,2,1)))

    conv3 = Convolution2D(params["n_filters_3"],
                          params["n_kernel_3"][0],
                          params["n_kernel_3"][1],
                          activation='relu',
                          init="uniform")
    x = conv3(x)
    logging.debug("Output Conv2D: %s" % str(conv3.output_shape))

    pool3 = MaxPooling2D(pool_size=(params["n_pool_3"][0],
                                    params["n_pool_3"][1]))
    x = pool3(x)
    logging.debug("Output MaxPool2D: %s" % str(pool3.output_shape))
    x = Dropout(params["dropout_factor"])(x)

    conv4 = Convolution2D(params["n_filters_4"],
                          params["n_kernel_4"][0],
                          params["n_kernel_4"][1],
                          activation='relu',
                          init="uniform")
    x = conv4(x)
    logging.debug("Output Conv2D: %s" % str(conv4.output_shape))

    pool4 = MaxPooling2D(pool_size=(params["n_pool_4"][0],
                                    params["n_pool_4"][1]))
    x = pool4(x)
    logging.debug("Output MaxPool2D: %s" % str(pool4.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    flat = Flatten()
    x = flat(x)
    logging.debug("Output Flatten: %s" % str(flat.output_shape))

    # dense1 = Dense(output_dim=params["n_dense"], init="uniform", activation='linear')
    # x = dense1(x)
    # logging.debug("Output CNN: %s" % str(dense1.output_shape))

    # metadata
    inputs2 = Input(shape=(params["n_metafeatures"],))
    dense2 = Dense(output_dim=params["n_dense"], init="uniform", activation='relu')
    x2 = dense2(inputs2)
    logging.debug("Output CNN: %s" % str(dense2.output_shape))

    x2 = Dropout(params["dropout_factor"])(x2)

    # merge
    xout = concatenate([x, x2], axis=1)

    dense3 = Dense(output_dim=output_size, init="uniform", activation='linear')
    xout = dense3(xout)
    logging.debug("Output CNN: %s" % str(dense3.output_shape))

    lambda1 = Lambda(lambda x: K.l2_normalize(x, axis=1))
    xout = lambda1(xout)

    model = Model(input=[inputs, inputs2], output=xout)

    return model


if __name__ == '__main__':
    pass
