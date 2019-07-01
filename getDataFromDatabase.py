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
		res.append(vocab_to_int.get(z, 1))
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
		ast = str2list(ast)
		for node in ast:
			if "type" in node.keys():
				counts[node["type"]] = counts.get(node["type"], 0) + 1
				vocab.update([node["type"]])
		# code2seq中path不包括value
		# 	if "value" in node.keys():
		# 	    counts[node["value"]] = counts.get(node["value"], 0) + 1
		# 	    vocab.update([node["value"]])

	_sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])
	for i, word in enumerate(["<PAD>", "<UNK>", "<START>", "<STOP>"] + _sorted):
		if vocab_size is not None and i > vocab_size:
			break

		vocab_to_int[word] = i

	return vocab_to_int


# def getSBT(ast, root):
# 	# 得到李戈的sbt树 （效果已经在多篇文章里证明不行了）
# 	cur_root = ast[root["index"]]
# 	tmp_list = []
# 	tmp_list.append("(")
# 	if "value" in cur_root.keys() and len(cur_root["value"]) > 0:
# 		str = cur_root["type"] + "_" + cur_root["value"] # 没有孩子
# 	else:
# 		str = cur_root["type"]
# 	tmp_list.append(str)
# 	if "children" in cur_root.keys():
# 		chs = cur_root["children"]
# 		for ch in chs:
# 			tmpl = getSBT(ast, ast[ch])
# 			tmp_list.extend(tmpl)
#
# 	tmp_list.append(")")
# 	return tmp_list


def getIndex(node):
	return node["index"]


# def str2list(ast):
# 	nodes = []
# 	while len(ast) > 0:
# 		idx = ast.find("},")
# 		if idx == -1:
# 			idx = ast.find("}")
# 		node = ast[:idx + 1]
#
# 		idx1 = node.find("type")
# 		if idx1 != -1:
# 			idx3 = node.find(",", idx1)
# 			if idx3 == -1:
# 				idx3 = node.find("}", idx1)
# 			type = node[idx1 + 6: idx3]
# 			new_type = '"' + type + '"'
# 			node = node[0: idx1 + 6] + new_type + node[idx3:]
# 		# node = node.replace(type, new_type)
#
# 		idx2 = node.find("value")
# 		if idx2 != -1:
# 			idx4 = node.find(",", idx2)
# 			if idx4 == -1:
# 				idx4 = len(node) - 1
# 			# idx4 = node.find("}", idx2)
# 			value = node[idx2 + 7: idx4]
# 			new_value = '"' + value + '"'
# 			node = node[0: idx2 + 7] + new_value + node[idx4:]
# 		# node = node.replace(value, new_value)
# 		nodes.append(json.loads(node))
# 		# print(node)
#
# 		if idx + 2 > len(ast):
# 			break
# 		ast = ast[idx + 3:]
# 	return sorted(nodes, key=getIndex)

def str2list(ast):
	nodes = []
	ast = json.loads(ast)
	for a in ast:
		nodes.append(a)
	return sorted(nodes, key=getIndex)


def getVocab():
	# 获得几种特征的词表
	# ast是json格式 n是需要抽取的路径数
	connect = pymysql.Connect(
		host="10.131.252.198",
		port=3306,
		user="root",
		passwd="17210240114",
		db="repos",
		charset='utf8'
	)

	cursor = connect.cursor()
	sql = "SELECT id, methName, tokens, comments, newapiseq FROM reposfile"
	# sql = "SELECT tokens FROM reposfile_new"
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

		# tokens.append(str(data[i][0]))
		# ast = str(data[i][1])
		# 这一步替换注
		# ast = str2list(ast)
		# asts.append(ast)
	length = len(data)
	del data
	cf = configs.conf()
	print(length)

	methName_vocab_to_int, methName_int_to_vocab = getVocabForOther(methNames, cf.n_words)
	token_vocab_to_int, token_int_to_vocab = getVocabForOther(tokens, cf.n_words)
	desc_vocab_to_int, desc_int_to_vocab = getVocabForOther(descs, cf.n_words)
	apiseq_vocab_to_int, apiseq_int_to_vocab = getVocabForOther(apiseqs, cf.n_words)  # api词表大小和其他的不一样
	# import pickle
	# methName_vocab_to_int = pickle.load(open(cf.data_dir + cf.vocab_methname, 'rb'))
	# token_vocab_to_int = pickle.load((open(cf.data_dir + cf.vocab_tokens, 'rb')))
	# desc_vocab_to_int = pickle.load((open(cf.data_dir + cf.vocab_desc, 'rb')))
	# apiseqnew_vocab_to_int = pickle.load((open(cf.data_dir + cf.vocab_apiseq, 'rb')))
	# 以上这些特征可以转为编号后重新写入数据库
	methNamesNum = []
	for methName in methNames:
		methNamesNum.append(toNum(methName, methName_vocab_to_int))

	tokensNum = []
	for token in tokens:
		tokensNum.append(toNum(token, token_vocab_to_int))

	descsNum = []
	for desc in descs:
		descsNum.append(toNum(desc, desc_vocab_to_int))

	apiseqsNum = []
	for apiseq in apiseqs:
		apiseqsNum.append(toNum(apiseq, apiseq_vocab_to_int))


	assert len(methNamesNum) == len(tokensNum) == len(descsNum) == len(apiseqsNum)
	sql = "insert INTO repos_index values (%s,%s,%s,%s,%s)"
	failed = 0
	for i in range(length):
		m, t, d, a = list2int(methNamesNum[i]), list2int(tokensNum[i]), list2int(descsNum[i]),  list2int(apiseqsNum[i])
		try:
			cursor.execute(sql, (ids[i], m, t, d, a))
		except Exception as e:
			print(e)
			print("insert failed")
			failed += 1
	cursor.close()
	connect.commit()
	connect.close()
	print("insert failed number is: %d" % failed)

	# ast_vocab_to_int = getVocabForAST(asts, cf.n_words-1)
	#
	# # ast的词表保存在本地
	# save_vocab("vocab_ast.json", ast_vocab_to_int)
	save_vocab("vocab_tokens.json", token_vocab_to_int)
	save_vocab("vocab_methname.json", methName_vocab_to_int)
	save_vocab("vocab_desc.json", desc_vocab_to_int)
	save_vocab("vocab_apiseq.json", apiseq_vocab_to_int)


def list2int(list):
	return " ".join([str(x) for x in list])


def save_vocab(path, params):
	with open(path, 'w') as f:
		json.dump(params, f)


def load_vocab(path):
	with open(path, 'r') as f:
		return json.load(f)


getVocab()
# ast_vocab_to_int = load_vocab("vocab_ast.json")
