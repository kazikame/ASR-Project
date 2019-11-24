import sys
from tensorflow.keras.models import load_model
from getError import evaluate
import numpy as np
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 getError.py model.h5")
		exit(-1)
	model = load_model(sys.argv[1])
	wmf_predictions, wmf_order, predictions, pred_order = evaluate(model)
	# print(wmf_predictions.shape, wmf_order.shape, predictions.shape, pred_order.shape)
	print(predictions.shape)
	valid_indices = np.loadtxt('valid_indices.txt').astype(int)
	np.random.seed(42)
	np.random.shuffle(valid_indices)
	actual_pred_indices = valid_indices[7000 + pred_order]
	ground_pred_indices = valid_indices[7000 + wmf_order]
	user_id_to_index = json.load(open('userToIndex.json'))
	song_id_to_index = json.load(open('songidToIndex.json'))
	user_play_dict = {}
	with open('topUsers.txt', 'r') as f:
		for line in f:
			user, song, count = line.split()
			if user not in user_id_to_index:
				continue
			if song not in song_id_to_index:
				continue
			if user not in user_play_dict:
				user_play_dict[user] = {song: int(count)}
			else:
				user_play_dict[user][song] = int(count)

	song_index_to_id = {v:k for k, v in song_id_to_index.items()}
	# user_index_to_id = {v:k for k, v in id_to_index.items()}
	id_to_name = json.load(open('id_to_name.json'))
	count1 = []
	size = actual_pred_indices.shape[1]
	for x in range(size):
		temp = 0
		for user in user_play_dict.keys():
			index = actual_pred_indices[user_id_to_index[user], -1*x]
			count = user_play_dict[user].get(song_index_to_id[index], 0)
			if count:
				# print(user, id_to_name[song_index_to_id[index]], count)
				temp += 1
		count1.append(temp)

	# rev_list = np.zeros(wmf_order.shape)
	# rows, cols = wmf_order.shape
	#
	# for i in range(rows):
	# 	for j in range(cols):
	# 		rev_list[i, wmf_order[i, j]] = j

	print()
	print()
	count2 = []
	assert (size == ground_pred_indices.shape[1])
	for x in range(size):
		temp = 0
		for user in user_play_dict.keys():
			index = ground_pred_indices[user_id_to_index[user], -1*x]
			count = user_play_dict[user].get(song_index_to_id[index], 0)
			if count:
				# print(user, id_to_name[song_index_to_id[index]], count)
				temp += 1
		count2.append(temp)
	plt.plot(count1)
	# plt.plot(count2)
	plt.show()
