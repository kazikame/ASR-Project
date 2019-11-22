import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dropout, MaxPooling2D, Convolution2D, Input, Lambda, concatenate, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.optimizers import Adam
from tensorflow import train as tftrain

import numpy as np
import pickle
import sys
TESTDATA = 7000
DIM = (84, 324, 1)


def evaluate(model):
	with open('pca.pickle', 'rb') as f:
		bow_complete = pickle.load(f)
	item_factors_complete = np.load('item_factors.npy')
	user_factors = np.load('user_factors.npy')
	valid_indices = np.loadtxt('valid_indices.txt').astype(int)
	testdata_indices = valid_indices[TESTDATA:]

	testdataSize = valid_indices.shape[0] - TESTDATA
	bow_testdata = np.empty((testdataSize, bow_complete.shape[1]))
	songs = np.zeros((testdataSize, *DIM))

	for i, ID in enumerate(testdata_indices):
		# Store sample
		temp = np.moveaxis(np.load('cqt_npy/' + str(ID) + '.npy'), 0, -1)
		songs[i, :, :temp.shape[1], :] = np.real(temp)
		# Store class
		bow_testdata[i, ] = bow_complete[ID, :]
		item_factors = item_factors_complete[ID, :]

	wmf_predictions = user_factors @ item_factors.T
	wmf_order = np.argsort(wmf_predictions, axis=1)

	predicted_factors = model.predict(songs, bow_testdata)
	predictions = user_factors @ predicted_factors.T
	pred_order = np.argsort(predictions, axis=1)
	return wmf_order, pred_order


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 getError.py model.h5")
		exit(-1)
	model = load_model(sys.argv[1])
	evaluate(model)
