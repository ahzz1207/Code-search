import random
import json

import pymysql


def getAllLeaves(ast):  # ast: list of nodes(dict)
	# ast 按照index排序
	# 给所有节点增加parent
	for node in ast:
		if "children" in node.keys():
			for childid in node["children"]:
				ast[childid]["parent"] = node["index"]

	# 获得所有的叶节点
	leaves = []
	for node in ast:
		if "children" not in node.keys():
			leaves.append(node)

	return ast, leaves


def getPathBetweenSrcTrg(src, trg, ast):
	# src, trg: node
	# ast: 有parent的ast
	srcAncestor = []
	trgAncestor = []
	ancestor = src
	while "parent" in ancestor.keys():
		srcAncestor.append(ast[ancestor["index"]])
		ancestor = ast[ancestor["parent"]]
	srcAncestor.append(ast[ancestor["index"]])

	ancestor = trg
	while "parent" in ancestor.keys():
		trgAncestor.append(ast[ancestor["index"]])
		ancestor = ast[ancestor["parent"]]
	trgAncestor.append(ast[ancestor["index"]])

	srcAncestor.reverse()
	trgAncestor.reverse()

	i = 0
	j = 0
	while i < len(srcAncestor) and j < len(trgAncestor):
		if srcAncestor[i]["index"] != trgAncestor[j]["index"]:
			break
		i += 1
		j += 1

	path = []
	srcToken = src["value"] if src["value"] != None else "None"
	trgToken = trg["value"] if trg["value"] != None else "None"

	i -= 1
	while i < len(srcAncestor):
		path.append(srcAncestor[i]["type"])
		i += 1
	path.reverse()

	while j < len(trgAncestor):
		path.append(trgAncestor[j]["type"])
		j += 1

	path = " ".join(path)
	return path, srcToken, trgToken


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


def parseInput(sent):
	return [z for z in sent.split(' ')]


def toNum(data, vocab_to_int):
	# 转为编号表示
	res = []
	for z in parseInput(data):
		res.append(str(vocab_to_int.get(z, vocab_to_int["<UNK>"])))
	return res


def astToNum(data, vocab_to_int):
	# 转为编号表示
	res = []
	for z in parseInput(data):
		res.append(vocab_to_int.get(z, vocab_to_int["<UNK>"]))
	if len(res) < 7:
		for i in range(7 - len(res)):
			res.append(vocab_to_int["<PAD>"])
	if len(res) > 7:
		res = res[:7]

	return res


def subtokenToNum(data, vocab_to_int):
	res = []
	for z in data:
		res.append(vocab_to_int.get(z, vocab_to_int["<UNK>"]))
	if len(res) < 3:
		for i in range(3 - len(res)):
			res.append(vocab_to_int["<PAD>"])
	if len(res) > 3:
		res = res[:3]

	return res


def getVocabForAST(asts, vocab_size):
	vocab = set()
	counts = {}

	vocab_to_int = {}
	int_to_vocab = {}

	for ast in asts:
		for node in ast:
			if "type" in node.keys():
				counts[node["type"]] = counts.get(node["type"], 0) + 1
				vocab.update([node["type"]])
			# # naive方法中path包括value
			# if "value" in node.keys():
			# 	counts[node["value"]] = counts.get(counts[node["value"]], 0) + 1
			# 	vocab.update(node["value"])

	_sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])
	for i, word in enumerate(["<PAD>", "<UNK>", "<START>", "<STOP>"] + _sorted):
		if vocab_size is not None and i > vocab_size:
			break

		vocab_to_int[word] = i
		int_to_vocab[i] = word

	return vocab_to_int, int_to_vocab


def dfsSimplify(ast, root, path, totalpath):
	# 深度遍历 得到多条路径
	if "children" in ast[root["index"]].keys():
		path.append(root["type"])
		for child in root["children"]:
			dfsSimplify(ast, ast[child], path, totalpath)
		path.pop()
	else:
		if root["value"] == None:
			root["value"] = "None"
		path.append(root["value"])
		# 叶节点内容包含在path中
		totalpath.append(' '.join(path))
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


def getPathSimplify(asts, pathNum, ast_vocab_to_int):
	# 每次训练路径都是随机抽取的
	astPathNum = []  # 所有ast的所有path的编号表示 三维数组
	for ast in asts:
		ast = json.loads(ast)
		nPath = getNPathSimplify(ast)  # 针对每个ast的所有路径
		nPathNum = []
		for path in nPath:  # 每条path的编号表示
			nPathNum.append(astToNum(path, ast_vocab_to_int))
		astPathNum.append(nPathNum)
	return astPathNum


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

# conn = pymysql.Connect(
# 			host="10.131.252.198",
# 			port=3306,
# 			user="root",
# 			passwd="17210240114",
# 			db="repos",
# 			charset='utf8'
# 		)
# cursor = conn.cursor()





def astindex():
	ast_vocab_to_int = json.load(open("./vocab_ast.json", 'r'))
	tokens_vocab_to_int = json.load(open("./vocab_tokens.json", 'r'))
	tokens_vocab_to_int = updateVocab(tokens_vocab_to_int)

	sql = "select ast from reposfile"
	cursor.execute(sql)
	conn.commit()
	data = cursor.fetchall()
	print(len(data))
	import tqdm
	indexdata = json.load(open('index.json', 'r'))
	print(len(indexdata))
	ids = json.load(open('ids.json', 'r'))
	j = 0
	valid = []
	validindex = json.load(open('valid.json', 'r'))
	with open('test.txt', 'w')as f:
		for index, row in tqdm.tqdm(enumerate(data)):
			ast = json.loads(row[0])
			astWithParent, leaves = getAllLeaves(ast)
			path = set()
			path_copy = []
			ftoken = []
			ltoken = []
			rspath = []
			rsl = []
			rsf = []
			for i in range(len(leaves) - 1):
				for j in range(i + 1, len(leaves)):
					pathSrcToTrg, srcToken, trgToken = getPathBetweenSrcTrg(leaves[i], leaves[j], ast)
					if pathSrcToTrg not in path:
						path.add(pathSrcToTrg)
						path_copy.append(pathSrcToTrg)
						ftoken.append(srcToken)
						ltoken.append(trgToken)
					if len(path) > 100:
						break
				if len(path) > 100:
					break
			for i in range(len(path)):
				rspath.append(astToNum(path_copy[i], ast_vocab_to_int))
				rsf.append(str(tokens_vocab_to_int.get(ftoken[i], 1)))
				rsl.append(str(tokens_vocab_to_int.get(ltoken[i], 1)))
			if index not in ids:
				f.write(json.dumps([rspath, ' '.join(rsf), ' '.join(rsl), indexdata[index]])+'\n')
			else:
				valid.append(json.dumps([rspath, ' '.join(rsf), ' '.join(rsl), validindex[j]]) + '\n')
				j += 1
	f.close()
	open('astvalid.json', 'w').writelines(valid)


def getindex(ast, tokens_vocab_to_int):
	ast = json.loads(ast)
	ast_vocab_to_int = json.load(open("./vocab_ast.json", 'r'))
	tokens_vocab_to_int = updateVocab(tokens_vocab_to_int)
	astWithParent, leaves = getAllLeaves(ast)
	path = set()
	path_copy = []
	ftoken = []
	ltoken = []
	rspath = []
	rsl = []
	rsf = []
	for i in range(len(leaves) - 1):
		for j in range(i + 1, len(leaves)):
			pathSrcToTrg, srcToken, trgToken = getPathBetweenSrcTrg(leaves[i], leaves[j], ast)
			if pathSrcToTrg not in path:
				path.add(pathSrcToTrg)
				path_copy.append(pathSrcToTrg)
				ftoken.append(srcToken)
				ltoken.append(trgToken)
			if len(path) > 100:
				break
		if len(path) > 100:
			break
	for i in range(len(path)):
		rspath.append(astToNum(path_copy[i], ast_vocab_to_int))
		rsf.append(tokens_vocab_to_int.get(ftoken[i], 1))
		rsl.append(tokens_vocab_to_int.get(ltoken[i], 1))
	return [rspath, rsf, rsl]

# 	npath = getNPathSimplify(ast) # 拿到所有路径
# 	npaths = []
# 	firstTokens = []
# 	lastTokens = []
# 	for path in npath:
# 		path = path.split(' ')
# 		npaths.append(astToNum(' '.join(path[1: -1]), ast_vocab_to_int))
# 		splitFirstToken = splitToken(path[0])
# 		firstTokens.append(subtokenToNum(splitFirstToken, tokens_vocab_to_int))
# 		splitLastToken = splitToken(path[-1])
# 		lastTokens.append(subtokenToNum(splitLastToken, tokens_vocab_to_int))
# 	cursor.execute(sql2, (json.dumps(npaths), json.dumps(firstTokens), json.dumps(lastTokens), row[0]))
#
# conn.commit()
