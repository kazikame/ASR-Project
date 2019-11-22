import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow import keras
import json
import pickle

pca_dim = 224
with open('tfidf.pickle', 'rb') as f:
	tf_idf_vector, tfidf_transformer, file_train_info, file_test_info, counts1, counts2 = pickle.load(f)

with open("../songidToIndex.json") as json_file:
	songind = json.load(json_file)

with open("tosongs.json") as json_file:
	tracktosong = json.load(json_file)

song_factors = np.load('../item_factors.npy')
y = np.zeros([len(file_train_info), song_factors.shape[1]])
countsnew = np.zeros([len(file_train_info), 5000])
counts1 = counts1.todense()
for trackid, (_, countind) in file_train_info.items():
	y[countind, :] = song_factors[songind[tracktosong[trackid][0]]]
	countsnew[songind[tracktosong[trackid][0]], :] = counts1[countind, :]

countsnew = csr_matrix(countsnew)
# model = keras.Sequential([
#     keras.layers.Dense(80, input_shape=pca_dim, activation='relu'),
#     keras.layers.Dense(song_factors.shape[1], activation='relu')
# ])


# model.compile(optimizer='adam',
#               loss= keras.losses.cosine_proximity,
#               metrics=['accuracy'])

svd = TruncatedSVD(n_components=pca_dim, n_iter=10, random_state=42)
svd.fit(tf_idf_vector)
pca = svd.transform(countsnew)
with open("pca.pickle", 'wb') as f:
	pickle.dump(pca, f)
# model.fit(svd.transform(counts1), y, epochs=10)
