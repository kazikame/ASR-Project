
from tensorflow.keras.models import load_model
from scipy.stats.stats import pearsonr

import numpy as np
import pickle
import sys
TESTDATA = 7000
DIM = (84, 324, 1)


def evaluate(model):
	with open('tfidf/pca.pickle', 'rb') as f:
		bow_complete = pickle.load(f)
	item_factors_complete = np.load('item_factors_128.npy')
	user_factors = np.load('user_factors_128.npy')
	valid_indices = np.loadtxt('valid_indices.txt').astype(int)
	testdata_indices = valid_indices[TESTDATA:]

	testdataSize = valid_indices.shape[0] - TESTDATA
	bow_testdata = np.empty((testdataSize, bow_complete.shape[1]))
	item_factors = np.empty((testdataSize, item_factors_complete.shape[1]))
	songs = np.zeros((testdataSize, *DIM))

	for i, ID in enumerate(testdata_indices):
		# Store sample
		temp = np.moveaxis(np.load('cqt_npy/' + str(ID) + '.npy'), 0, -1)
		songs[i, :, :temp.shape[1], :] = np.real(temp)
		# Store class
		bow_testdata[i, ] = bow_complete[ID, :]
		item_factors[i, ] = item_factors_complete[ID, :]

	wmf_predictions = user_factors @ item_factors.T
	wmf_order = np.argsort(-1*wmf_predictions, axis=1)
	predicted_factors = model.predict([songs, bow_testdata])
	predictions = user_factors @ predicted_factors.T
	pred_order = np.argsort(-1*predictions, axis=1)
	assert(wmf_order.shape == pred_order.shape)
	return wmf_predictions, wmf_order, predictions, pred_order

def corrcoef_loss(wmf_order, pred_order):
	rows = wmf_order.shape[0]
	ans = [pearsonr(wmf_order[i, :], pred_order[i, :])[0] for i in range(rows)]
	return np.array(ans)

def inversion_loss(wmf_order, pred_order):
	rev_list = np.zeros(wmf_order.shape)
	rows, cols = wmf_order.shape
	for i in range(rows):
		for j in range(cols):
			rev_list[i, wmf_order[i, j]] = j

	inv_count = 0
	for i in range(rows):
		for x in range(cols):
			for y in range(x+1, cols):
				if (rev_list[i, pred_order[i, x]] > rev_list[i, pred_order[i, y]]):
					inv_count += 1

	return inv_count / wmf_order.size



if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 getError.py model.h5")
		exit(-1)
	model = load_model(sys.argv[1])
	_, wmf_order, _, pred_order = evaluate(model)
	print("Avg. Correlation Coeff:", np.mean(corrcoef_loss(wmf_order, pred_order)))
	print("Avg. number of inversions:", np.mean(inversion_loss(wmf_order, pred_order)))
