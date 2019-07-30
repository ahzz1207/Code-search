import pymysql
import json
import redis
import tqdm
import random
r = redis.Redis(charset='utf-8')
f = open('astindex.txt', 'r')
data = []
index = 0
for row in tqdm.tqdm(f.readlines()):
	index += 1
	data.append(json.loads(row.strip()))
	if index == 500000:
		random.shuffle(data)
		index = 0
		for line in data:
			r.rpush('index', json.dumps(line[:5]))
		data = []
		print("Done epoch:", r.llen('index'))
print("ALLＤＯＮＥ")


