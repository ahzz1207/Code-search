import numpy as np
from tensorflow.python.keras.layers import Concatenate, Dot, Embedding, Lambda, Activation, Dense, Dropout, Layer
from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.engine import Input
from tensorflow.python.keras import backend as K
from tensorflow.python import keras
from tensorflow.python.keras import optimizers
import tensorflow as tf
import os
import transformer
import h5py


class JointEmbeddingModel:
	def __init__(self, config):
		self.data_dir = config.data_dir
		self.model_name = config.model_name
		self.methname_len = config.methname_len  # the max length of method name
		self.apiseq_len = config.apiseq_len
		self.tokens_len = config.tokens_len
		self.desc_len = config.desc_len
		self.ast_words = config.ast_words
		self.vocab_size = config.n_words  # the size of vocab
		self.embed_dims = config.embed_dims
		self.lstm_dims = config.lstm_dims
		self.hidden_dims = config.hidden_dims
		self.astpath_len = config.astpath_len
		self.astpath_num = config.path_num
		self.node_words = config.node_words
		self.batch_size = config.batch_size
		self.margin = 0.05

		self.init_embed_weights_methodname = config.init_embed_weights_methodname
		self.init_embed_weights_tokens = config.init_embed_weights_tokens
		self.init_embed_weights_desc = config.init_embed_weights_desc

		self.methodname = Input(shape=(self.methname_len,), batch_size=self.batch_size, dtype='int32', name='methodname')
		# self.apiseq = Input(shape=(self.apiseq_len,), dtype='int32', name='apiseq')
		# self.tokens = Input(shape=(self.tokens_len,), dtype='int32', name='tokens')
		self.desc_good = Input(shape=(self.desc_len,), batch_size=self.batch_size, dtype='int32', name='desc_good')
		self.desc_bad = Input(shape=(self.desc_len,), batch_size=self.batch_size, dtype='int32', name='desc_bad')
		self.astpath = Input(shape=(self.astpath_num, self.astpath_len), batch_size=self.batch_size, dtype='int32', name='astpaths')
		self.firstNode = Input(shape=(self.astpath_num, 5), batch_size=self.batch_size, dtype='int32', name='firstNodes')
		self.lastNode = Input(shape=(self.astpath_num, 5), batch_size=self.batch_size, dtype='int32', name='lastNodes')

		# self.nodeEmbed = embedingNode(self.embed_dims, self.node_words)

		# create path to store model Info
		if not os.path.exists(self.data_dir + 'model/' + self.model_name):
			os.makedirs(self.data_dir + 'model/' + self.model_name)

	def build(self):
		# 1 -- CodeNN
		methodname = Input(shape=(self.methname_len,), batch_size=self.batch_size, dtype='int32', name='methodname')
		# apiseq = Input(shape=(self.apiseq_len,), dtype='int32', name='apiseq')
		# tokens = Input(shape=(self.tokens_len,), dtype='int32', name='tokens')
		##
		astpath = Input(shape=(self.astpath_num, self.astpath_len,), batch_size=self.batch_size, dtype='int32', name='astpaths')
		firstNode = Input(shape=(self.astpath_num, 5,), batch_size=self.batch_size, dtype='int32', name='firstNodes')
		lastNode = Input(shape=(self.astpath_num, 5,), batch_size=self.batch_size, dtype='int32', name='lastNodes')

		astpaths = tf.reshape(astpath, shape=(self.batch_size*self.astpath_num, self.astpath_len))  # batch*220, ast_len
		firstNodes = tf.reshape(firstNode, shape=(self.batch_size, self.astpath_num * 5))
		lastNodes = tf.reshape(lastNode, shape=(self.batch_size, self.astpath_num * 5))
		##

		self.transformer_meth = transformer.EncoderModel(vocab_size=self.vocab_size, model_dim=self.lstm_dims,
		                                                 embed_dim=self.embed_dims, ffn_dim=self.lstm_dims,
		                                                 droput_rate=0.2, n_heads=8, max_len=self.methname_len,
		                                                 name='methT')

		self.transformer_ast = transformer.EncoderModel(vocab_size=self.ast_words, model_dim=self.lstm_dims,
		                                                   embed_dim=self.embed_dims, ffn_dim=self.lstm_dims,
		                                                   droput_rate=0.3, n_heads=8, max_len=self.astpath_len,
		                                                   name='astT')

		self.transformer_desc = transformer.EncoderModel(vocab_size=self.vocab_size, model_dim=self.lstm_dims,
		                                                 embed_dim=self.embed_dims, ffn_dim=self.lstm_dims,
		                                                 droput_rate=0.2, n_heads=8, max_len=self.desc_len,
		                                                 name='descT')



		# 3 ast

		nodembedding = Embedding(
			input_dim=self.node_words,
			output_dim=self.embed_dims,
			weights=None,
			mask_zero=False,
			name='embedding_Node'
		)

		astpath_out = self.transformer_ast(astpaths)
		#
		first_mask = tf.not_equal(firstNodes, 0)
		first_mask = tf.tile(tf.expand_dims(first_mask, axis=2), (1, 1, self.embed_dims))
		last_mask = tf.not_equal(lastNodes, 0)
		last_mask = tf.tile(tf.expand_dims(last_mask, axis=2), (1, 1, self.embed_dims))

		first_embed = nodembedding(firstNodes)
		last_embed = nodembedding(lastNodes)
		zeros = tf.zeros_like(first_embed)
		first_embed = tf.where(first_mask, first_embed, zeros)
		last_embed = tf.where(last_mask, last_embed, zeros)

		## AST
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_ast')
		astpath_out = maxpool(astpath_out)  # batch*220, hidden
		ast_out = tf.reshape(astpath_out, shape=(self.batch_size, self.astpath_num, self.hidden_dims))

		## subtoken
		first_embed = tf.reshape(first_embed, shape=(-1, self.astpath_num, 5, self.embed_dims))
		last_embed = tf.reshape(last_embed, shape=(-1, self.astpath_num, 5, self.embed_dims))
		sumpool = Lambda(lambda x: K.sum(x, axis=2, keepdims=False), output_shape=lambda x: (x[0], x[1], x[3]),
		                 name='sumpooling_node')
		first_sumpool = sumpool(first_embed)
		last_sumpool = sumpool(last_embed)

		## concat
		astpath_concat = Concatenate(axis=2, name='astpathConcat')(
			[ast_out, first_sumpool, last_sumpool])
		astpath_out = Dense(self.lstm_dims, 'tanh', name='astpath')(astpath_concat)

		meanpool = Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                  name='meanpooling_astpaths')
		astpath_repr = meanpool(astpath_out)
		#

		# 4 methodname
		# embedding layer
		meth_name_out = self.transformer_meth(methodname)
		# max pooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_methodname')
		method_name_pool = maxpool(meth_name_out)
		activation = Activation('tanh', name='active_method_name')
		methodname_repr = activation(method_name_pool)


		# fusion methodname, apiseq, tokens
		merge_ast_repr = Concatenate(name='merge_methname_ast')([methodname_repr, astpath_repr])
		merge_ast_repr = Dense(units=self.hidden_dims, name="code_repr", activation="tanh")(merge_ast_repr)

		code_repr = merge_ast_repr
		# merge_methname_api = Concatenate(name='merge_methname_api')([merge_ast_repr, ])
		# merge_code_repr = Concatenate(name='merge_code_repr')([merge_ast_repr, tokens_repr])

		# code_repr = Dense(4*self.hidden_dims, activation='tanh', name='dense_coderepr')(merge_ast_repr)

		self.code_repr_model = Model(inputs=[methodname, astpath, firstNode, lastNode],
		                             outputs=[code_repr], name='code_repr_model')
		self.code_repr_model.summary()

		#  2 -- description
		desc = Input(shape=(self.desc_len,), batch_size=self.batch_size, dtype='int32', name='desc')

		# desc
		# embedding layer
		desc_out = self.transformer_desc(desc)

		# max pooling

		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_desc')
		desc_pool = maxpool(desc_out)
		activation = Activation('tanh', name='active_desc')
		desc_repr = activation(desc_pool)

		self.desc_repr_model = Model(inputs=[desc], outputs=[desc_repr], name='desc_repr_model')
		self.desc_repr_model.summary()

		# 3 -- cosine similarity
		# code_repr = self.code_repr_model([methodname, astpath, firstNode, lastNode])
		# desc_repr = self.desc_repr_model([desc])
		#
		# cos_sim = Dot(axes=1, normalize=True, name='cos_sim')([code_repr, desc_repr])
		# sim_model = Model(inputs=[methodname, astpath, firstNode, lastNode, desc], outputs=[cos_sim], name='sim_model')
		#
		# self.sim_model = sim_model
		#
		# self.sim_model.summary()

		# 4 -- build training model
		# good_sim = sim_model([self.methodname, self.astpath, self.firstNode, self.lastNode, self.desc_good])
		# bad_sim = sim_model([self.methodname, self.astpath, self.firstNode, self.lastNode, self.desc_bad])
		code_repr_v = self.code_repr_model([self.methodname, self.astpath, self.firstNode, self.lastNode])
		good_desc_repr = self.desc_repr_model([self.desc_good])
		bad_desc_repr = self.desc_repr_model([self.desc_bad])
		good_sim = Dot(axes=1, normalize=True, name='cos_sim_good')([code_repr_v, good_desc_repr])
		bad_sim = Dot(axes=1, normalize=True, name='cos_sim_bad')([code_repr_v, bad_desc_repr])
		loss = Lambda(lambda x: K.maximum(1e-6, self.margin - x[0] + x[1]), output_shape=lambda x: x[0], name='loss')(
			[good_sim, bad_sim])

		self.training_model = Model(inputs=[self.methodname, self.astpath, self.firstNode, self.lastNode,
		                                    self.desc_good, self.desc_bad],
		                            outputs=[loss], name='training_model')

		self.training_model.summary()

	def compile(self, optimizer, **kwargs):
		optimizer = optimizers.Adam(lr=0.0001)
		self.code_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
		self.desc_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
		self.training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=optimizer, **kwargs)
		# self.sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

	def fit(self, x, **kwargs):
		y = np.zeros(shape=x[0].shape[:1], dtype=np.float32)
		return self.training_model.fit(x, y, **kwargs)

	def repr_code(self, x, **kwargs):
		return self.code_repr_model.predict(x, **kwargs)

	def repr_desc(self, x, **kwargs):
		return self.desc_repr_model.predict(x, **kwargs)

	def predict(self, x, **kwargs):
		# return self.sim_model.predict(x, **kwargs)
		pre = Dot(axes=1, normalize=True, name="cos_sim")([self.repr_code(x[:4]), self.repr_desc(x[-1])])
		return np.array(pre).flatten()

	def save(self, code_model_file, desc_model_file, **kwargs):
		# self.code_repr_model.save_weights(code_model_file, **kwargs)
		# self.desc_repr_model.save_weights(desc_model_file, **kwargs)
		file = h5py.File(code_model_file, 'w')
		weight_code = self.code_repr_model.get_weights()
		for i in range(len(weight_code)):
			file.create_dataset('weight_code' + str(i), data=weight_code[i])
		file.close()

		file = h5py.File(desc_model_file, 'w')
		weight_desc = self.desc_repr_model.get_weights()
		for i in range(len(weight_desc)):
			file.create_dataset('weight_desc' + str(i), data=weight_desc[i])
		file.close()

	def load(self, code_model_file, desc_model_file, **kwargs):
		# self.code_repr_model.load_weights(code_model_file, **kwargs)
		# self.desc_repr_model.load_weights(desc_model_file, **kwargs)
		file = h5py.File(code_model_file, 'r')
		weight_code = []
		for i in range(len(file.keys())):
			weight_code.append(file['weight_code' + str(i)][:])
		self.code_repr_model.set_weights(weight_code)
		file.close()

		file = h5py.File(desc_model_file, 'r')
		weight_desc = []
		for i in range(len(file.keys())):
			weight_desc.append(file['weight_desc' + str(i)][:])
		self.desc_repr_model.set_weights(weight_desc)
		file.close()

