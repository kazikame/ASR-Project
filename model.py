from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dropout, MaxPooling2D, Convolution2D, Input, Lambda, concatenate, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.optimizers import Adam

params = {
    'dropout_factor': 0.5,
    'n_dense': 1024,
    'n_dense_2': 2048,
    'n_filters_1': 256,
    'n_filters_2': 512,
    'n_filters_3': 1024,
    'n_filters_4': 1024,
    'n_kernel_1': (84, 4),
    'n_kernel_2': (1, 4),
    'n_kernel_3': (1, 4),
    'n_kernel_4': (1, 1),
    'n_out': '',
    'n_pool_1': (1, 4),
    'n_pool_2': (1, 4),
    'n_pool_3': (1, 1),
    'n_pool_4': (1, 1),
    'n_pool_5': (1, 1),
    'n_frames': 322,
    'n_mel': 96,
    'architecture': 2,
    'batch_norm': False,
    'dropout': True
}


def get_model(input_shape1, input_shape2, output_size):
    inputs = Input(shape=input_shape1)

    conv1 = Convolution2D(params["n_filters_1"], params["n_kernel_1"][0],
                          params["n_kernel_1"][1],
                          padding='same',
                          activation='relu')
    x = conv1(inputs)
    # print("Input CNN: %s" % str(inputs.output_shape))
    print("Output Conv2D: %s" % str(conv1.output_shape))

    pool1 = MaxPooling2D(pool_size=(params["n_pool_1"][0],
                                    params["n_pool_1"][1]))
    x = pool1(x)
    print("Output MaxPool2D: %s" % str(pool1.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    conv2 = Convolution2D(params["n_filters_2"], params["n_kernel_2"][0],
                          params["n_kernel_2"][1],
                          padding='same',
                          activation='relu')
    x = conv2(x)
    print("Output Conv2D: %s" % str(conv2.output_shape))

    pool2 = MaxPooling2D(pool_size=(params["n_pool_2"][0],
                                    params["n_pool_2"][1]))
    x = pool2(x)
    print("Output MaxPool2D: %s" % str(pool2.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    # model.add(Permute((3,2,1)))

    conv3 = Convolution2D(params["n_filters_3"],
                          params["n_kernel_3"][0],
                          params["n_kernel_3"][1],
                          activation='relu')
    x = conv3(x)
    print("Output Conv2D: %s" % str(conv3.output_shape))

    pool3 = MaxPooling2D(pool_size=(params["n_pool_3"][0],
                                    params["n_pool_3"][1]))
    x = pool3(x)
    print("Output MaxPool2D: %s" % str(pool3.output_shape))
    x = Dropout(params["dropout_factor"])(x)

    conv4 = Convolution2D(params["n_filters_4"],
                          params["n_kernel_4"][0],
                          params["n_kernel_4"][1],
                          activation='relu')
    x = conv4(x)
    print("Output Conv2D: %s" % str(conv4.output_shape))

    pool4 = MaxPooling2D(pool_size=(params["n_pool_4"][0],
                                    params["n_pool_4"][1]))
    x = pool4(x)
    print("Output MaxPool2D: %s" % str(pool4.output_shape))

    x = Dropout(params["dropout_factor"])(x)

    flat = Flatten()
    x = flat(x)
    print("Output Flatten: %s" % str(flat.output_shape))

    dense = Dense(params["n_dense"], activation='linear')
    x = dense(x)
    # print("Output CNN: %s" % str(dense1.output_shape))

    # metadata
    inputs2 = Input(shape=input_shape2)
    dense1 = Dense(params["n_dense"], activation='relu')
    x2 = dense1(inputs2)
    dense2 = Dense(params["n_dense_2"], activation='relu')
    x2 = dense2(x2)
    print("Output CNN: %s" % str(dense2.output_shape))

    x2 = Dropout(params["dropout_factor"])(x2)

    # merge
    xout = concatenate([x, x2], axis=1)
    # print(xout.shape)
    dense3 = Dense(output_size, activation='linear')
    xout = dense3(xout)
    print("Output CNN: %s" % str(dense3.output_shape))

    lambda1 = Lambda(lambda x: K.l2_normalize(x, axis=1))
    xout = lambda1(xout)

    model = Model(inputs=[inputs, inputs2], outputs=xout)
    opt = Adam(lr=0.01)
    model.compile(loss=cosine_similarity, optimizer=opt, metrics=['mse'])

    return model


if __name__ == '__main__':
    pass
