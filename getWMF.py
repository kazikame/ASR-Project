import implicit
import numpy as np
import json
import sys
import json
from scipy import sparse
import os
NUM_FACTORS = 1024

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def convertToCSR(filename, num_users, num_songs, user_hash={}, song_hash={}):
    array = np.zeros((num_songs, num_users))
    # user_hash = {}
    # song_hash = {}
    c_user = 0
    c_song = 0

    if os.path.isfile('itemsongcsr.npz'):
        print("CSR file already exists, skipping!")
        csr = sparse.load_npz('itemsongcsr.npz')
        print(csr.todense())
        return csr
    print("Making numpy array")
    with open(filename, 'r') as f:
        for line in f:
            user, songid, count = line.split()
            # print("User:", user)
            if user in user_hash and songid in song_hash:
                # print("Done for user=", user, user_hash[user], "song = ", songid, song_hash[songid])
                array[song_hash[songid], user_hash[user]] = int(count)

    print("Numpy array made")
    with open('songidToIndex.json', 'w') as f:
        json.dump(song_hash, f)
    with open('userToIndex.json', 'w') as f:
        json.dump(user_hash, f)
    print("Indexes made")
    csr = sparse.csr_matrix(array)
    sparse.save_npz('itemsongcsr.npz', csr)
    print("CSR matrix made")
    return csr


def getWMF(csr):
    model = implicit.als.AlternatingLeastSquares(factors=NUM_FACTORS, regularization=0.01, dtype=np.float32, use_native=True, use_cg=True, iterations=30, calculate_training_loss=True)
    model.fit(csr, show_progress=True)
    item_factors = model.item_factors
    user_factors = model.user_factors
    np.save('user_factors.npy', user_factors)
    np.save('item_factors.npy', item_factors)
    print(np.sum(~item_factors.any(1)))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 getWMF.py topUsers.txt num_users num_items")
        exit(-1)

    user_hash = {}
    k = 0
    with open('dataset_generation/userlist.txt', 'r') as f:
        for line in f:
            user_hash[line.strip()] = k
            k += 1

    song_hash = {}
    k = 0
    with open('dataset_generation/songlist.txt', 'r') as f:
        for line in f:
            song_hash[line.split('<SEP>')[1]] = k
            k += 1

    print("Hashes generated")
    # print(user_hash)
    # exit()
    # print(song_hash)
    getWMF(convertToCSR(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), user_hash, song_hash))

