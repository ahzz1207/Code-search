import pymysql
import json
import redis
import tqdm
import random
r = redis.Redis(charset='utf-8')
f = open('astvalid.txt', 'r')
data = []
i = 0
for row in tqdm.tqdm(f.readlines()):
	data.append(json.loads(row))
random.shuffle(data)
for line in data:
	length = len(line[4])
	while len(line[4]) < 100:
		i = random.randrange(0, length)
		line[4].append(line[4][i])
		line[5].append(line[5][i])
		line[6].append(line[6][i])
	r.rpush('astvalid', json.dumps(line))





# for row in tqdm.tqdm(f.readlines()):
# 	i += 1
# 	data.append(json.loads(row))
# 	if i % 1000000 == 0:
# 		random.shuffle(data)
# 		for line in tqdm.tqdm(data):
# 			length = len(line[4])
# 			while len(line[4]) < 100:
# 				i = random.randrange(0, length)
# 				line[4].append(line[4][i])
# 				line[5].append(line[5][i])
# 				line[6].append(line[6][i])
# 			r.rpush('astindex', json.dumps(line))
# 		data = []
# 		print("Done epoch:", r.llen('onlyindex'))
#
# random.shuffle(data)
# for line in tqdm.tqdm(data):
# 	length = len(line[4])
# 	while len(line[4]) < 100:
# 		i = random.randrange(0, length)
# 		line[4].append(line[4][i])
# 		line[5].append(line[5][i])
# 		line[6].append(line[6][i])
# 	r.rpush('astindex', json.dumps(line))
# data = []
# print("Done epoch:", r.llen('onlyindex'))
# print("ALLＤＯＮＥ")


