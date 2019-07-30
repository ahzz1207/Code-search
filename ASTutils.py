import copy
import random
import json

import pymysql


def parseInput(sent):
	return [z for z in sent.split(' ')]


def toNum(data, vocab_to_int):
	# 转为编号表示
	res = []
	for z in parseInput(data):
		res.append(str(vocab_to_int.get(z, 1)))
	return res


def tokenToNum(data, vocab_to_int):
	# 转为编号表示
	res = []
	for z in parseInput(data):
		res.append(str(vocab_to_int.get(z, 1)))
	if len(res) > 3:
		res = res[:3]
	elif len(res) < 3:
		for i in range(3 - len(res)):
			res.append(0)
	return res


def astToNum(data, vocab_to_int):
	# 转为编号表示
	res = []

	if len(data) < 7:
		for i in range(7 - len(data)):
			data.append("<PAD>")
	elif len(data) > 7:
		data = data[:7]

	for z in (data):
		res.append(vocab_to_int.get(z, vocab_to_int["<UNK>"]))

	return res


def dfsSimplify(ast, root, path, totalpath):
	# 深度遍历 得到多条路径
	if "children" in ast[root["index"]].keys():
		path.append(root["type"])
		for child in root["children"]:
			dfsSimplify(ast, ast[child], path, totalpath)
		path.pop()
	else:
		# todo
		if root["value"] == None:
			root["value"] = "None"
		path.append(root["value"])
		# 叶节点内容包含在path中
		totalpath.append(' '.join(path))
		if len(totalpath) > 200:
			return
		path.pop()
		return


def getNPathSimplify(ast):
	# 随机得到n条路径
	path = []
	totalpath = []
	dfsSimplify(ast, ast[0], path, totalpath)
	nPath = []
	n = len(totalpath)
	for i in range(n):
		for j in range(i + 1, n):
			sent = ' '.join(reversed(totalpath[i].split(' ')[1:])) + ' ' + totalpath[j]
			nPath.append(sent)

	return nPath


# def getPathSimplify(asts, pathNum, ast_vocab_to_int):
# # 	# 每次训练路径都是随机抽取的
# # 	astPathNum = []  # 所有ast的所有path的编号表示 三维数组
# # 	firstToNum = []
# # 	lastTokNum = []
# # 	for ast in asts:
# # 		ast = json.loads(ast)
# # 		nPath = getNPathSimplify(ast, pathNum)  # 针对每个ast的n条路径
# # 		nPathNum = []
# # 		firstNum = []
# # 		lastNum = []
# # 		for path in nPath:  # 每条path的编号表示
# # 			firstNum.append(tokenToNum(path[0]), tokens_vocab_to_int)
# # 			lastNum.append(tokenToNum(path[-1]), tokens_vocab_to_int)
# # 			nPathNum.append(astToNum(path, ast_vocab_to_int))
# # 		astPathNum.append(nPathNum)
# # 		firstToNum.append(firstNum)
# # 		lastTokNum.append(lastNum)
# # 	return astPathNum


def splitToken(token):
	subtokens = []
	start = 0
	l = len(token)
	end = start + 1
	while start < l:
		sb = ""
		sb += token[start].lower()
		end = start + 1
		while end < l and token[end].islower():
			sb += token[end]
			end += 1
		subtokens.append(sb)
		if end < l:
			start = end
		else:
			break

	return subtokens


def subtokenToNum(data, vocab_to_int):
	if len(data) < 3:
		for i in range(3 - len(data)):
			data.append("<PAD>")
	elif len(data) > 3:
		data = data[:3]
	res = []
	for z in data:
		res.append(vocab_to_int.get(z, 1))
	return res


def updateVocab(vocab):
	size = len(vocab)
	keyWord = {
		"abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "continue",
		"default", "do", "double", "else", "enum", "exports", "extends", "final", "finally", "float", "for",
		"if", "implements", "import", "instanceof", "int", "interface", "long", "long", "module", "native", "new",
		"package", "private", "protected", "public", "requires", "return", "short", "static", "strictfp", "super",
		"switch", "synchronized",
		"this", "throw", "throws", "transient", "try", "void", "volatile", "while", "true", "null", "false", "var",
		"const", "goto"
	}
	keyWord = list(keyWord)
	stopwords = {
		"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yourself", "yourselves",
		"he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
		"theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
		"are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
		"a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
		"about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
		"up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
		"when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
		"no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
		"should", "now", "lbrace", "rbrace", "dot", "comma", "eq", "semi", "lparen", "rparen", "colon", "lbracket",
		"rbracket",
		"lt", "gt", "{", "}", "(", ")", "[", "]", ",", "."
	}
	stopwords = list(stopwords)
	for i in range(len(keyWord)):
		vocab.update([(keyWord[i], size + i)])
	size = len(vocab)
	for i in range(len(stopwords)):
		vocab.update([(stopwords[i], size + i)])

	return vocab

conn = pymysql.Connect(
			host="10.131.252.198",
			port=3306,
			user="root",
			passwd="17210240114",
			db="repos",
			charset='utf8'
		)
cursor = conn.cursor()



# sql = "select reponame from repos_deal"
# set = set()
# cursor.execute(sql)
# for row in cursor.fetchall():
# 	set.add(row[0])
# with open("addrlist.txt") as f:
# 	for line in f.readlines():
# 		temp = line.strip().split("/")
# 		name = temp[-2] + "$$%" + temp[-1]
# 		set.add(name)
#
# addr = []
# data = json.load(open("response.json"))
# for id in data["data"]:
# 	temp = id["local_addr"].split("/")
# 	name = temp[-2] + "$$%" + temp[-1]
# 	if name not in set:
# 		addr.append(id["local_addr"] + "\n")
# print(len(addr))
# with open("addrlist.txt", "w") as f:
# 	f.writelines(addr)



# pre = "public class test%d {\n %s \n}"
# rs = []
# count = 0
# temp = []
# guolv = set()
# with open("C:\\Users\\loading\\Desktop\\SAGA-GPU-fragment\\type12_frag_result.csv", 'r', encoding='utf-8') as f:
# 	for row in tqdm(f.readlines()):
# 		if row == "\n":
# 			rs.append(count)
# 			rs.append(copy.copy(temp))
# 			count = 0
# 			temp.clear()
# 			guolv.clear()
# 		else:
# 			if row.split(',')[1] not in guolv:
# 				count += 1
# 				temp.append(row)
# 				guolv.add(row.split(',')[1])
# 	f.close()
# with open("D:\\result2.csv", "w") as f:
# 	for row in tqdm(rs):
# 		if type(row) == list:
# 			f.writelines(row)
# 		else:
# 			f.write(str(row) + '\n')
# 	f.close()

# print(pre % (1, "public void register(Job job){   jobs.add(job); } "))


def getAstVocab():
	ast_vocab_to_int = json.load(open("vocab_ast.json", 'r'))
	tokens_vocab_to_int = json.load(open("vocab_tokens.json", 'r'))
	tokens_vocab_to_int = updateVocab(tokens_vocab_to_int)
	import tqdm
	sql = "select id, ast from reposfile where id < 1000"
	cursor.execute(sql)
	conn.commit()
	data = cursor.fetchall()
	with open("astindex.json", "w")as f:
		for row in tqdm.tqdm(data):
			rs = []
			npath = getNPathSimplify(json.loads(row[1]))
			npaths = []
			nfirst = []
			nlast = []
			for path in npath:
				path = path.split(' ')
				nfirst.append(subtokenToNum(splitToken(path[0]), tokens_vocab_to_int))
				nlast.append(subtokenToNum(splitToken(path[-1]), tokens_vocab_to_int))
				npaths.append(astToNum(path[1:-1], ast_vocab_to_int))
				rs.append([nfirst, nlast, npaths])
			del row
			f.write(json.dumps(rs))
		f.close()


# def getRandom():
# 	dic = json.load(open("ids.json", 'r'))
# 	data = json.load(open('index.json', 'r'))
# 	with open('test.txt', 'r') as f:
# 		for id, line in enumerate(f.readlines()):
			

getRandom()

# tokens_vocab_to_int = updateVocab(tokens_vocab_to_int)
# idindex = json.load(open('idlist.txt', 'r'))
# idlist = []
# sql = "select id, astindex from repos_index"
# cursor.execute(sql)
# conn.commit()
# data = cursor.fetchall()
# for row in data:
# 	if row[1]:
# 		astindex = json.loads(row[1])
# 		for path in astindex:
# 			if len(path) != 9:
# 				idlist.append(row[0])
# 				print(row[0])
# 				break
# 	else:
# 		idlist.append(row[0])
# 	print(row[0])
# print(len(idlist))
# with open('idlist.txt', 'w') as f:
# 	json.dump(idlist, f)



# clones = {}
# with open("C:\\Users\\loading\\Desktop\\SAGA-GPU-fragment\\new_result.csv", 'r') as f:
# 	for line in f.readlines():
# 		if ',' not in line:
# 			id = int(line.rstrip('\n'))
# 			if id not in clones:
# 				clones[id] = 1
# 			else:
# 				clones[id] += 1
# 	sorte = sorted(clones.keys())
# 	rs = {}
# 	for i in sorte:
# 		rs[i] = clones[i]
# 	json.dump(rs, open("clones.json", 'w'))
# 	f.close()
#
# with open("C:\\Users\\loading\\Desktop\\SAGA-GPU-fragment\\new_result.csv", 'w') as f:
# 	for dub in rs:
# 		if type(dub) == list:
# 			f.writelines(dub)
# 		else:
# 			f.write(str(dub))
# 			f.write('\n')
# 	f.close()



# import numpy as np
# import matplotlib.pyplot as plt
# with open("clones.json", 'r') as f:
# 	xy = json.load(f)
# 	x=[]
# 	y=[]
# 	count = 0
# 	tcount = 0
# 	for i in xy.keys():
# 		id = int(i)
# 		if id > 20:
# 			tcount += xy[i]
# 		count += xy[i]
# 		x.append(int(i))
# 		y.append(xy[i])
# 	print(tcount, count)
#
# 	plt.xticks(np.linspace(1, 100, 10))
# 	plt.bar(x[2:100], y[2:100], 0.25, color='b')
# 	plt.xlabel('group size')
# 	plt.ylabel('group number')
# 	# plt.hist(y, x, histtype='bar')
# 	plt.savefig("clones.png")
# 	plt.show()
