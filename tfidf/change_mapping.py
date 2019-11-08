import json
songmap={}
with open('unique_tracks.txt') as f:
	for line in f:
		fields= line.strip().split('<SEP>')
		songmap[fields[0]]= (fields[1], fields[2], fields[3])

with open("tosongs.json", 'w') as json_file:
    json.dump(songmap, json_file)