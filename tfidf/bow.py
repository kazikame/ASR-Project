import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import json
mxmids=set()
file_train_info={}
file_test_info={}
ind =-1
count1=0
count2= 0
numrow=0
tfidf_songs={}
with open('../songidToIndex.json', 'r') as f:
	songind = json.load(f)

downloaded_ids= set()
with open('songtorest.json', 'r') as f:
	totrack= json.load(f)
	for key in songind.keys():
		downloaded_ids.add(totrack[key][0])

# print(downloaded_ids)
counts1 = []
with open('mxm_dataset_train.txt', 'r') as f:
	for idx, line in enumerate(f):
		if line[0] == '%' or line[0] == '#':
			continue
		if ind == -1:
			ind= idx
		mxid = line.split(',', 1)[0]
		if mxid in downloaded_ids:
			python_dict = literal_eval('{' + ",".join(line.strip().split(",", 2)[2:]) + '}')
			temp= np.zeros(5000)
			for key, val in python_dict.items():
				temp[int(key)-1]=int(val)
			file_train_info[mxid] = (idx, count1)
			count1 += 1
			# print(temp.shape)
			counts1.append(temp)
		numrow += 1

counts1= np.vstack(counts1)
counts1= csr_matrix(counts1)
counts2= []
with open('mxm_dataset_test.txt', 'r') as f:
	for idx, line in enumerate(f):
		if line[0] == '%' or line[0] == '#':
			continue
		mxid = line.split(',', 1)[0]
		if mxid in downloaded_ids:
			python_dict = literal_eval('{'+ ",".join(line.strip().split(",", 2)[2:])+'}')
			temp= np.zeros(5000)
			for key, val in python_dict.items():
				temp[int(key)-1]=int(val)
			file_test_info[mxid]=(idx, count2)
			count2 += 1
			counts2.append(temp)
# counts2= np.vstack(counts2)
# counts2= csr_matrix(counts2)
counts=np.zeros([int(numrow), 5000])

with open('mxm_dataset_train.txt', 'r') as f:
	for idx, line in enumerate(f):
		if line[0]=='%' or line[0]=='#':
			continue
		python_dict = literal_eval('{'+ ",".join(line.strip().split(",", 2)[2:])+'}')
		mxid=line.strip().split(",", 1)[0]
		for key, val in python_dict.items():
			counts[idx-ind][int(key)-1]=int(val)

counts= csr_matrix(counts)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(counts)
# reqcounts= np.zeros([len(intersected_ids), 5000])
tf_idf_vector=tfidf_transformer.transform(counts)
with open('tfidf.pickle', 'wb') as f:
	# pickle.dump([tf_idf_vector, tfidf_transformer, file_train_info, file_test_info, tfidf_transformer.transform(counts1), tfidf_transformer.transform(counts2)], f)
	pickle.dump([tf_idf_vector, tfidf_transformer, file_train_info, file_test_info, tfidf_transformer.transform(counts1), "hello"], f)

