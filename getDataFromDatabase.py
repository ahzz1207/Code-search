import pymysql
import json
import re
import random
import configs
from operator import itemgetter
import collections


def parseInput(sent):
	return [z for z in sent.split(' ')]


def toNum(data, vocab_to_int):
	# 转为编号表示
	res = []
	for z in parseInput(data):
		index = vocab_to_int.get(z, -1)
		if index == -1:
			# 假定UNK是1
			res.append(1)
		else:
			res.append(index)
	return res


def getVocabForOther(datas, vocab_size):
	# 为其他特征生成词表
	vocab = set()
	counts = {}

	vocab_to_int = {}
	int_to_vocab = {}

	for data in datas:
		words = parseInput(data)
		for word in words:
			counts[word] = counts.get(word, 0) + 1
		vocab.update(words)

	_sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])
	for i, word in enumerate(["<PAD>", "<UNK>", "<START>", "<STOP>"] + _sorted):
		if vocab_size is not None and i > vocab_size:
			break

		vocab_to_int[word] = i
		int_to_vocab[i] = word

	return vocab_to_int, int_to_vocab


def getVocabForAST(asts, vocab_size):
	# 为ast的type和value生成词表 获得所有的type和value
	vocab = set()
	counts = {}

	vocab_to_int = {}
	int_to_vocab = {}

	for ast in asts:
		for node in ast:
			if "type" in node.keys():
				counts[node["type"]] = counts.get(node["type"], 0) + 1
				vocab.update([node["type"]])
			# code2seq中path不包括value
			# if "value" in node.keys():
			# 	counts[node["value"]] = counts.get(node["value"], 0) + 1
			# 	vocab.update([node["value"]])

	_sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])
	for i, word in enumerate(["<PAD>", "<UNK>", "<START>", "<STOP>"] + _sorted):
		if vocab_size is not None and i > vocab_size:
			break

		vocab_to_int[word] = i
		int_to_vocab[i] = word

	return vocab_to_int, int_to_vocab


def dfs(ast, root, path, totalpath):
	# 深度遍历 得到多条路径
	if "children" in ast[root["index"]].keys():
		path.append(root["type"])
		for child in root["children"]:
			dfs(ast, ast[child], path, totalpath)
		path.pop()
	else:
		if root["value"] == None:
			root["value"] = "None"
		path.append(root["value"])
		# code2seq中叶节点内容不包含在path中 而是subtoken
		totalpath.append(' '.join(path))
		path.pop()
		return


def getNPath(ast, n):
	# 随机得到n条路径
	path = []
	totalpath = []
	dfs(ast, ast[0], path, totalpath)
	nPath = []
	for i in range(n):
		a = random.randint(0, len(totalpath) - 1)
		b = random.randint(0, len(totalpath) - 1)
		sent = ' '.join(reversed(totalpath[a].split(' ')[1:])) + ' ' + totalpath[b]
		nPath.append(sent)
	return nPath


def getIndex(node):
	return node["index"]


def str2list(ast):
	nodes = []
	ast = json.loads(eval(ast))
	for a in ast:
		nodes.append(a)
	return sorted(nodes, key=getIndex)


def getVocab():
	# 获得几种特征的词表
	# ast是json格式 n是需要抽取的路径数
	connect = pymysql.Connect(
		host="localhost",
		port=3306,
		user="root",
		passwd="Taylorswift-1997",
		db="githubreposfile",
		charset='utf8'
	)
	cursor = connect.cursor()
	# sql = "SELECT id, methName, tokens, comments, apiseq, newapiseq FROM reposfile_new"
	sql = "SELECT * FROM reposfile"
	cursor.execute(sql)
	data = cursor.fetchall()

	asts = []
	methNames = []
	tokens = []
	descs = []
	apiseqs = []
	newapiseq = []

	# todo: 绑定id
	ids = []
	for i in range(len(data)):
		ids.append(int(data[i][0]))
		methName = str(data[i][1])
		methNames.append(methName)

		token = str(data[i][2])
		tokens.append(token)

		desc = str(data[i][3])
		descs.append(desc)

		apiseq = str(data[i][4])
		apiseqs.append(apiseq)
		#
		# if data[i][5]:
		# 	apiseqnew = str(data[i][5])
		# 	newapiseq.append(apiseqnew)
		# else:
		# 	newapiseq.append("%%%%")

		ast = str(data[i][6])
		ast = str2list(ast)
		asts.append(ast)
	length = len(data)
	data = []
	cf = configs.conf()
	print(length)

	methName_vocab_to_int, methName_int_to_vocab = getVocabForOther(methNames, cf.n_words)
	token_vocab_to_int, token_int_to_vocab = getVocabForOther(tokens, cf.n_words)
	desc_vocab_to_int, desc_int_to_vocab = getVocabForOther(descs, cf.n_words)
	apiseq_vocab_to_int, apiseq_int_to_vocab = getVocabForOther(apiseqs, cf.api_words)  # api词表大小和其他的不一样
	# apiseqnew_vocab_to_int, apiseqnew_int_to_vocab = getVocabForOther(newapiseq, cf.new_api_words)

	# 以上这些特征可以转为编号后重新写入数据库
	# methNamesNum = []
	# for methName in methNames:
	# 	methNamesNum.append(toNum(methName, methName_vocab_to_int))
	#
	# tokensNum = []
	# for token in tokens:
	# 	tokensNum.append(toNum(token, token_vocab_to_int))
	#
	# descsNum = []
	# for desc in descs:
	# 	descsNum.append(toNum(desc, desc_vocab_to_int))
	#
	# apiseqsNum = []
	# for apiseq in apiseqs:
	# 	apiseqsNum.append(toNum(apiseq, apiseq_vocab_to_int))
	#
	# apiseqnewNum = []
	# for apiseqn in newapiseq:
	# 	if apiseqn == "%%%%":
	# 		apiseqnewNum.append([-10])
	# 	else:
	# 		apiseqnewNum.append(toNum(apiseqn, apiseqnew_vocab_to_int))
	#
	#
	# assert len(methNamesNum) == len(tokensNum) == len(descsNum) == len(apiseqsNum)
	# sql = "INSERT INTO repos2index values (%s,%s,%s,%s,%s,%s)"
	# failed = 0
	# for i in range(length):
	# 	m, t, d, a, q = list2int(methNamesNum[i]), list2int(tokensNum[i]), list2int(descsNum[i]), list2int(apiseqsNum[i]), list2int(apiseqnewNum[i])
	# 	try:
	# 		cursor.execute(sql, (ids[i], m, t, d, a, q))
	# 	except Exception as e:
	# 		print(e)
	# 		print("insert failed")
	# 		failed += 1
	# cursor.close()
	# connect.commit()
	# connect.close()
	# print("insert failed number is: %d" % failed)

	ast_vocab_to_int, ast_int_to_vocab = getVocabForAST(asts, cf.n_words)
	getPath(asts, 10, ast_vocab_to_int, token_vocab_to_int)

	# ast的词表保存在本地
	save_vocab("vocab_ast.json", ast_vocab_to_int)
	save_vocab("vocab_method.json", methName_vocab_to_int)
	save_vocab("vocab_api.json", apiseq_vocab_to_int)
	save_vocab("vocab_token.json", token_vocab_to_int)
	save_vocab("vocab_desc.json", desc_vocab_to_int)


def list2int(list):
	return " ".join([str(x) for x in list])


def save_vocab(path, params):
	with open(path, 'w') as f:
		json.dump(params, f)


def load_vocab(path):
	with open(path, 'r') as f:
		return json.load(f)


def getPath(asts, pathNum, ast_vocab_to_int, token_vocab_to_int):
	# 每次训练路径都是随机抽取的
	astPathNum = []  # 所有ast的所有path的编号表示 三维数组
	firstTokenNum = []
	lastTokenNum = []
	for ast in asts:
		nPath = getNPath(ast, pathNum)  # 针对每个ast的n条路径
		nPathNum = []
		firstToken = []
		lastToken = []
		for path in nPath:  # 每条path的编号表示
			path = path.split(' ')
			nPathNum.append(toNum(' '.join(path[1:-1]), ast_vocab_to_int))
			firstToken.append(toNum(path[0], token_vocab_to_int))
			lastToken.append(toNum(path[-1], token_vocab_to_int))
		astPathNum.append(nPathNum)
		firstTokenNum.append(firstToken)
		lastTokenNum.append(lastToken)
	# sbt = ' '.join(getSBT(ast, ast[0]))  # 得到李戈的sbt树
	return astPathNum


getVocab()
# ast_vocab_to_int = load_vocab("vocab_ast.json")
