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

	if len(data) < 9:
		for i in range(9 - len(data)):
			data.append("<PAD>")
	elif len(data) > 9:
		data = data[:9]

	for z in (data):
		res.append(vocab_to_int.get(z, vocab_to_int["<UNK>"]))

	return res


def dfsSimplify(ast, root, path, totalpath):
	# 深度遍历 得到多条路径
	if "children" in ast[root["index"]].keys():
		if len(ast[root["index"]]["children"]) >= 1:
			path.append(root["type"])
			for child in root["children"]:
				dfsSimplify(ast, ast[child], path, totalpath)
			path.pop()
		else:
			# 只有一个子节点 略过
			dfsSimplify(ast, ast[root["children"][0]], path, totalpath)
	else:
		# todo
		if root["value"] == None:
			root["value"] = "None"
		path.append(root["value"])
		# 叶节点内容包含在path中
		totalpath.append(' '.join(path))
		return


def getNPathSimplify(ast, n):
	# 随机得到n条路径
	path = []
	totalpath = []
	dfsSimplify(ast, ast[0], path, totalpath)
	nPath = []
	for i in range(n):
		a = random.randint(0, len(totalpath) - 1)
		b = random.randint(0, len(totalpath) - 1)
		sent = ' '.join(reversed(totalpath[a].split(' ')[1:])) + ' ' + totalpath[b]
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
	if len(data) < 5:
		for i in range(5 - len(data)):
			data.append("<PAD>")
	elif len(data) > 5:
		data = data[:5]
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


from tqdm import tqdm
pre = "public class test%d {\n %s \n}"
with open("E:\\methods.json", 'r', encoding='utf-8') as f:
	for row in tqdm(f.readlines()):
		try:
			js = json.loads(row)
			# cont = ";\n".join(row[1].split(";"))
			cont = js["method"]
			id = js["mid"]
			code = pre % (id, cont)
			filenmae = "d:\\code2\\" + str(id) + ".java"
			with open(filenmae, 'w') as o:
				o.write(code)
				o.close()
		except Exception as e:
			print(e)

# print(pre % (1, "public void register(Job job){   jobs.add(job); } "))

# ast_vocab_to_int = json.load(open("vocab_ast.json", 'r'))
# tokens_vocab_to_int = json.load(open("vocab_tokens.json", 'r'))
#
# tokens_vocab_to_int = updateVocab(tokens_vocab_to_int)
#
# sql = "select id, ast from reposfile where id = 3527218"
# sql2 = " update repos_index set astindex = %s, firstindex = %s, lastindex = %s where id = %s "
# cursor.execute(sql)
# conn.commit()
# data = cursor.fetchall()
#
# for row in data:
# 	npath = getNPathSimplify(json.loads(row[1]), 220)
# 	npaths = []
# 	nfirst = []
# 	nlast = []
# 	for path in npath:
# 		path = parseInput(path)
# 		nfirst.append(subtokenToNum(splitToken(path[0]), tokens_vocab_to_int))
# 		nlast.append(subtokenToNum(splitToken(path[-1]), tokens_vocab_to_int))
# 		npaths.append(astToNum(path[1:-1], ast_vocab_to_int))
# 	cursor.execute(sql2, (json.dumps(npaths), json.dumps(nfirst), json.dumps(nlast), row[0]))
# 	conn.commit()

# tokens_vocab_to_int = updateVocab(tokens_vocab_to_int)
# idindex = json.load(open('idlist.txt', 'r'))

# idindex = []
# sql = "select id, astindex from repos_index"
# sql2 = "update repos_index set astindex = %s where id = %s "
# cursor.execute(sql)
# conn.commit()
# data = cursor.fetchall()
# for row in data:
# 	if row[1]:
# 		ast = json.loads(row[1])
# 		for path in ast:
# 			if len(path) != 9:
# 				idindex.append(row[0])
# 				break
# 	else:
# 		idindex.append(row[0])
# 	print(row[0])
# json.dump(idindex, open('idindex.txt', 'w'))

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
