import tqdm
import json
import copy

comment = {}
rawcode = {}
with open("E:\\comments.json", 'r', encoding="utf-8")as f:
	for line in tqdm.tqdm(f.readlines()):
		row = json.loads(line.strip())
		comment[row["mid"]] = row["comment"]
	f.close()
print(len(comment.keys()))

with open("E:\\methods.json", 'r', encoding="utf-8")as f:
	for line in tqdm.tqdm(f.readlines()):
		row = json.loads(line.strip())
		rawcode[row["mid"]] = row["method"]
	f.close()
print(len(rawcode.keys()))

w = open("result_3to10.txt", 'w', encoding="utf-8")

with open("value_counts_3to10.txt", 'r', encoding="utf-8") as f:
	for line in f.readlines():
		sb = []
		sb.append(copy.deepcopy(line))
		line = line.strip().split("    ")
		id = line[0].split("\\")
		id = id[-1].split(".")
		sb.append(comment[int(id[0])]+"\n")
		sb.append(rawcode[int(id[0])]+"\n")
		w.writelines(sb)
	f.close()
w.close()
