import math
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)
import pickle
import tables
import configsA
import codecs
import random
from scipy.stats import rankdata
import traceback
import threading
from utils import normalize, cos_np_for_normalized
# from models_notoken import *
from model_tokens_path import *
import pymysql
import json
import numpy as np
import os
import redis

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class CodeSearcher:
	def __init__(self, conf):
		self.conf = conf
		self.path = self.conf.data_dir

		self.data_len = 0
		self.code_base = None
		self.code_base_chunksize = 1000000
		self.code_reprs = None
		self._eval_sets = None
		self.r = None

	def load_pickle(self, filename):
		return pickle.load(open(filename, 'rb'))


	def get_dataset2(self):
		self.r = redis.Redis(charset='utf-8')

		self.data_len = self.r.llen('index')

		print("All index init succes, it's length ：%d" % self.data_len)

	def get_valid_dataset(self):
		self.r = redis.Redis(charset='utf-8')

		self.data_len = self.r.llen('valid')

		print("All index init succes, it's length ：%d" % self.data_len)


	def load_hdf5(self, file, start_offset, chunk_size):
		table = tables.open_file(file)
		data, index = (table.get_node('/phrases'), table.get_node('/indices'))
		data_len = index.shape[0]
		if chunk_size == -1:  # load all data
			chunk_size = data_len
		start_offset = start_offset % data_len
		offset = start_offset
		sents = []
		while offset < start_offset + chunk_size:
			if offset >= data_len:
				chunk_size = start_offset + chunk_size - data_len
				start_offset = 0
				offset = 0
			len, pos = index[offset]['length'], index[offset]['pos']
			offset += 1
			sents.append(data[pos: pos + len].astype('int32'))
		table.close()
		return sents

	def load_train_data(self, start_offset, chunk_size, db):
		chunk_methnames = []
		chunk_descs = []
		chunk_asts = []
		chunk_tokens = []
		chunk_apiseq = []
		pad = [0 for _ in range(7)]
		start_offset = start_offset % self.data_len

		if start_offset + chunk_size > self.data_len:
			data = self.r.lrange(db, 0, chunk_size-1)
		else:
			data = self.r.lrange(db, start_offset, start_offset + chunk_size-1)

		for row in data:
			row = json.loads(row)
			chunk_methnames.append(np.array([int(x, base=10) for x in row[0].strip().split(' ')]))
			chunk_tokens.append(np.array([int(x, base=10) for x in row[1].strip().split(' ')]))
			chunk_descs.append(np.array([int(x, base=10) for x in row[2].strip().split(' ')]))
			chunk_apiseq.append(np.array([int(x, base=10) for x in row[3].strip().split(' ')]))
			if len(row[4]) < 100:
				row[4] += [pad for _ in range(100 - len(row[4]))]
			else:
				row[4] = row[4][:100]
			chunk_asts.append(row[4])
		chunk_padded_astspaths = np.asarray(chunk_asts)
		del data, chunk_asts
		return chunk_methnames, chunk_tokens, chunk_apiseq, chunk_descs, chunk_padded_astspaths


	def load_valid_data(self, chunk_size):
		chunk_methnames = self.load_hdf5(self.path + self.conf.valid_methodname, 0, chunk_size)
		chunk_apiseq = self.load_hdf5(self.path + self.conf.valid_apiseq, 0, chunk_size)
		chunk_tokens = self.load_hdf5(self.path + self.conf.valid_tokens, 0, chunk_size)
		chunk_descs = self.load_hdf5(self.path + self.conf.valid_desc, 0, chunk_size)
		return chunk_methnames, chunk_apiseq, chunk_tokens, chunk_descs

	def load_use_data(self):
		methnames = self.load_hdf5(self.path + self.conf.use_methodname, 0, -1)
		apiseq = self.load_hdf5(self.path + self.conf.use_apiseq, 0, -1)
		tokens = self.load_hdf5(self.path + self.conf.use_tokens, 0, -1)
		return methnames, apiseq, tokens

	def load_codebase(self):
		if self.code_base == None:
			code_base = []
			codes = codecs.open(self.path + self.conf.use_codebase, encoding='utf-8', errors='replace').readlines()
			print(len(codes))
			for i in range(0, len(codes), self.code_base_chunksize):
				code_base.append(codes[i: i + self.code_base_chunksize])
			self.code_base = code_base

	def load_code_reprs(self):
		if self.code_reprs == None:
			codereprs = []
			h5f = tables.open_file(self.conf.use_codevecs)
			vecs = h5f.root.vecs
			for i in range(0, len(vecs), self.code_base_chunksize):
				codereprs.append(vecs[i: i + self.code_base_chunksize])
			h5f.close()
			self.code_reprs = codereprs
		return self.code_reprs

	def save_code_reprs(self, vecs):
		npvecs = np.array(vecs)
		fvec = tables.open_file(self.conf.use_codevecs, 'w')
		atom = tables.Atom.from_dtype(npvecs.dtype)
		filters = tables.Filters(complib='blosc', complevel=5)
		ds = fvec.create_carray(fvec.root, 'vecs', atom, npvecs.shape, filters=filters)
		ds[:] = npvecs
		fvec.close()

	def convert(self, vocab, words):
		if type(words) == str:
			words = words.strip().lower().split(' ')
		return [vocab.get(w, 0) for w in words]

	def revert(self, vocab, indices):
		ivocab = dict((v, k) for k, v in vocab.items())
		return [ivocab.get(i, 'UNK') for i in indices]

	def pad(self, data, len=None):
		from tensorflow.python.keras.preprocessing.sequence import pad_sequences
		return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

	def save_model_epoch(self, model, epoch):
		if not os.path.exists(self.path + 'models-tokens+path/' + self.conf.model_name + '/'):
			os.makedirs(self.path + 'models-tokens+path/' + self.conf.model_name + '/')

		model.save("{}models-tokens+path/{}/epo{:d}_code.h5".format(self.path, self.conf.model_name, epoch),
		           "{}models-tokens+path/{}/epo{:d}_desc.h5".format(self.path, self.conf.model_name, epoch),
		           overwrite=True)

	def load_model_epoch(self, model, epoch):

		model.load("{}models-tokens+path/{}/epo{:d}_code.h5".format(self.path, self.conf.model_name, epoch),
		           "{}models-tokens+path/{}/epo{:d}_desc.h5".format(self.path, self.conf.model_name, epoch))
		print("Load model %epoch" % epoch)


	def train(self, model):
		codesearcher.get_dataset2()
		if self.conf.reload > 0:
			self.load_model_epoch(model, self.conf.reload)
		save_every = self.conf.save_every
		batch_size = self.conf.batch_size
		nb_epoch = self.conf.nb_epoch
		split = self.conf.validation_split
		val_loss = {'loss': 1., 'epoch': 0}

		for i in range(self.conf.reload, nb_epoch):
			print('Epoch %d' % i, end=' ')
			chunk_methnames, chunk_tokens, chunk_apiseq, chunk_descs, chunk_padded_astspaths \
				= self.load_train_data(i * self.conf.chunk_size, self.conf.chunk_size, 'index')

			chunk_padded_methnames = self.pad(chunk_methnames, self.conf.methname_len)
			chunk_padded_tokens = self.pad(chunk_tokens, self.conf.tokens_len)
			chunk_padded_apiseq = self.pad(chunk_apiseq, self.conf.apiseq_len)
			chunk_padded_good_descs = self.pad(chunk_descs, self.conf.desc_len)
			chunk_bad_descs = [desc for desc in chunk_descs]
			random.shuffle(chunk_bad_descs)
			chunk_padded_bad_descs = self.pad(chunk_bad_descs, self.conf.desc_len)

			inputs = [chunk_padded_apiseq, chunk_padded_tokens, chunk_padded_methnames, chunk_padded_astspaths, chunk_padded_good_descs, chunk_padded_bad_descs]
			hist = model.fit(x=inputs, epochs=1, batch_size=batch_size, validation_split=split)

			if hist.history['val_loss'][0] < val_loss['loss']:
				val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
				self.save_model_epoch(model, i)
			elif i % save_every == 0:
				self.save_model_epoch(model, i)
			print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))
			# if valid_every is not None and i % valid_every == 0:
			# 	acc1, mrr = self.valid(model, 1000, 1)
			# 	print(acc1, mrr)


	def valid(self, model, poolsize, K):

		data_len = poolsize
		#  poolsize - size of the code pool, if -1, load the whole test set
		c_1, c_2 = 0, 0
		for i in range(poolsize):
			bad_descs = [desc for desc in self._eval_sets['descs']]
			random.shuffle(bad_descs)
			descs = bad_descs
			descs[0] = self._eval_sets['descs'][i]  # good desc
			descs = self.pad(descs, self.conf.desc_len)
			methnames = self.pad([self._eval_sets['methnames'][i]] * data_len, self.conf.methname_len)
			apiseqs = self.pad([self._eval_sets['apiseqs'][i]] * data_len, self.conf.apiseq_len)
			tokens = self.pad([self._eval_sets['tokens'][i]] * data_len, self.conf.tokens_len)
			n_good = K

			sims = model.predict([methnames, apiseqs, tokens, descs], batch_size=data_len).flatten()
			r = rankdata(sims, method='max')
			max_r = np.argmax(r)
			max_n = np.argmax(r[:n_good])
			c_1 += 1 if max_r == max_n else 0
			c_2 += 1 / float(r[max_r] - r[max_n] + 1)

		top1 = c_1 / float(data_len)
		# percentage of predicted most similar desc that is really the corresponding desc
		mrr = c_2 / float(data_len)
		print("Top-1={}, MRR={}".format(top1, mrr))
		return top1, mrr

	def eval(self, model, poolsize, K):
		"""
        validate in a code pool.
        param:
            poolsize - size of the code pool, if -1, load the whole test set
        """

		def ACC(real, predict):
			sum = 0.0
			for val in real:
				try:
					index = predict.index(val)
				except ValueError:
					index = -1
				if index != -1: sum = sum + 1
			return sum / float(len(real))

		def MAP(real, predict):
			sum = 0.0
			for id, val in enumerate(real):
				try:
					index = predict.index(val)
				except ValueError:
					index = -1
				if index != -1: sum = sum + (id + 1) / float(index + 1)
			return sum / float(len(real))

		def MRR(real, predict):
			sum = 0.0
			for val in real:
				try:
					index = predict.index(val)
				except ValueError:
					index = -1
				if index != -1: sum = sum + 1.0 / float(index + 1)
			return sum / float(len(real))

		def NDCG(real, predict):
			dcg = 0.0
			idcg = IDCG(len(real))
			for i, predictItem in enumerate(predict):
				if predictItem in real:
					itemRelevance = 1
					rank = i + 1
					dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
			return dcg / float(idcg)

		def IDCG(n):
			idcg = 0
			itemRelevance = 1
			for i in range(n):
				idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
			return idcg

		# load valid dataset
		self.get_valid_dataset()
		acc, mrr, map, ndcg = 0, 0, 0, 0
		batch_size = self.conf.valid_batch_size
		chunk_methnames, chunk_tokens, chunk_apiseq, chunk_descs, chunk_padded_astspaths \
			= self.load_train_data(0, poolsize, 'valid')
		data_len = len(chunk_methnames)
		print("Eval dataSet length %d" % data_len, batch_size)
		for i in range(data_len):

			desc = chunk_descs[i]  # good desc
			descs = self.pad([desc] * data_len, self.conf.desc_len)
			methnames = self.pad(chunk_methnames, self.conf.methname_len)
			tokens = self.pad(chunk_tokens, self.conf.tokens_len)
			apiseqs = self.pad(chunk_apiseq, self.conf.apiseq_len)

			n_results = K
			sims = []
			for j in range(data_len // batch_size):
				inputs = [apiseqs[j*batch_size: (j+1)*batch_size], tokens[j*batch_size: (j+1)*batch_size],
				          methnames[j*batch_size: (j+1)*batch_size], chunk_padded_astspaths[j*batch_size: (j+1)*batch_size],
				          descs[j*batch_size: (j+1)*batch_size]]

				sim = model.predict(x=inputs, batch_size=batch_size)

				for x in sim:
					sims.append(x)

			negsims = np.negative(sims)
			predict = np.argsort(negsims)  # predict = np.argpartition(negsims, kth=n_results-1)
			predict = predict[:n_results]
			predict = [int(k) for k in predict]
			real = [i]
			acc += ACC(real, predict)
			mrr += MRR(real, predict)
			map += MAP(real, predict)
			ndcg += NDCG(real, predict)

			if i % 1000 == 0:
				print(i, acc, acc / float(i+1), mrr / float(i+1))

		acc = acc / float(data_len)
		mrr = mrr / float(data_len)
		map = map / float(data_len)
		ndcg = ndcg / float(data_len)

		return acc, mrr, map, ndcg

	def repr_code(self, model):
		methnames, apiseqs, tokens = self.load_use_data()
		methnames = self.pad(methnames, self.conf.methname_len)
		apiseqs = self.pad(apiseqs, self.conf.apiseq_len)
		tokens = self.pad(tokens, self.conf.tokens_len)
		vecs = model.repr_code([methnames, apiseqs, tokens], batch_size=1000)
		vecs = vecs.astype('float32')
		vecs = normalize(vecs)
		self.save_code_reprs(vecs)
		return vecs

	def search(self, model, query, n_results=10):
		desc = [self.convert(self.vocab_desc, query)]  # convert desc sentence to word indices
		padded_desc = self.pad(desc, self.conf.desc_len)
		desc_repr = model.repr_desc([padded_desc])
		desc_repr = desc_repr.astype('float32')

		codes = []
		sims = []
		threads = []
		for i, code_reprs_chunk in enumerate(self.code_reprs):
			t = threading.Thread(target=self.search_thread,
			                     args=(codes, sims, desc_repr, code_reprs_chunk, i, n_results))
			threads.append(t)
		for t in threads:
			t.start()
		for t in threads:  # wait until all sub-threads finish
			t.join()
		return codes, sims

	def search_thread(self, codes, sims, desc_repr, code_reprs, i, n_results):
		# 1. compute similarity
		chunk_sims = cos_np_for_normalized(normalize(desc_repr), code_reprs)
		print(chunk_sims.shape)
		# 2. choose top results
		negsims = np.negative(chunk_sims[0])
		maxinds = np.argpartition(negsims, kth=n_results - 1)
		maxinds = maxinds[:n_results]
		chunk_codes = [self.code_base[i][k] for k in maxinds]
		chunk_sims = chunk_sims[0][maxinds]
		codes.extend(chunk_codes)
		sims.extend(chunk_sims)

	def postproc(self, codes_sims):
		codes_, sims_ = zip(*codes_sims)
		codes = [code for code in codes_]
		sims = [sim for sim in sims_]
		final_codes = []
		final_sims = []
		n = len(codes_sims)
		for i in range(n):
			is_dup = False
			for j in range(i):
				if codes[i][:80] == codes[j][:80] and abs(sims[i] - sims[j]) < 0.01:
					is_dup = True
			if not is_dup:
				final_codes.append(codes[i])
				final_sims.append(sims[i])
		return zip(final_codes, final_sims)


if __name__ == '__main__':
	conf = configsA.conf()
	codesearcher = CodeSearcher(conf)
	mode = 'eval'
	#  Define model
	model = eval(conf.model_name)(conf)
	model.build()
	model.compile()

	if mode == 'train':
		codesearcher.train(model)

	elif mode == 'eval':
		# evaluate for a particular epoch
		# load model
		if conf.reload > 0:
			codesearcher.load_model_epoch(model, conf.reload)
		acc, mrr, map, ndcg = codesearcher.eval(model, 2000, 10)
		print("Eval result is :")
		print(acc, mrr, map, ndcg)

	elif mode == 'repr_code':
		# load model
		if conf.reload > 0:
			codesearcher.load_model_epoch(model, conf.reload)
		vecs = codesearcher.repr_code(model)


	elif mode == 'search':
		# search code based on a desc
		if conf.reload > 0:
			codesearcher.load_model_epoch(model, conf.reload)
		codesearcher.load_code_reprs()
		codesearcher.load_codebase()
		while True:
			try:
				query = input('Input Query: ')
				n_results = int(input('How many results? '))
			except Exception:
				print("Exception while parsing your input:")
				traceback.print_exc()
				break
			codes, sims = codesearcher.search(model, query, n_results)
			zipped = zip(codes, sims)
			zipped = sorted(zipped, reverse=True, key=lambda x: x[1])
			zipped = codesearcher.postproc(zipped)
			zipped = list(zipped)[:n_results]
			results = '\n\n'.join(map(str, zipped))  # combine the result into a returning string
			print(results)

