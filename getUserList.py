from collections import Counter

def userMatrix(song_list, user_file, output_file='topUsers.txt'):
	user_counter = Counter()
	song_counter = Counter()
	user_song_counter = {}
	songset = set(song_list)
	with open(user_file, 'r') as f:

		for line in f:
			username, songid, count = line.split()
			count = int(count)
			if songid in songset:
				if username not in user_song_counter:
					user_song_counter[username] = Counter()
				user_counter[username] += count
				song_counter[songid] += count
				user_song_counter[username][songid] = count

		
	with open(output_file, 'w') as f:
		for user in user_counter.most_common():
			for song in user_song_counter[user[0]]:
				count = user_song_counter[user[0]][song]
				print(user[0], song, count, file=f)


def getTopSongs(user_file, numTop=10000):
	song_counter = Counter()
	song_list = []
	with open(user_file, 'r') as f:
		for line in f:
			username, songid, count = line.split()
			count = int(count)
			song_counter[songid] += count

	for tup in song_counter.most_common(numTop):
		song_list.append(tup[0])

	with open('top_' + str(numTop)+'_songs.txt', 'w') as f:
		for i in song_list:
			print(i, song_counter[i], file=f)

	return song_list


if __name__ == '__main__':
	song_list = []

	with open('subset_unique_tracks.txt', 'r') as f:
		for line in f:
			_, songid, _, _ = line.split('<SEP>')
			song_list.append(songid)
	# song_list = getTopSongs('dataset/train_triplets.txt')
	userMatrix(song_list, 'train_triplets.txt')
