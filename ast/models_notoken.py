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


class embedingNode(Layer):
	def __init__(self, embed_dims, node_words):
		super(embedingNode, self).__init__(name='embed')
		nodembedding = self.add_weight(name='NodeEmbed', shape=(node_words, embed_dims), dtype=tf.float32,
		                                trainable=True)
		self.lookup = tf.concat((tf.zeros(shape=(1, embed_dims), dtype=tf.float32), nodembedding[1:, :]), axis=0)

	def call(self, inputs, **kwargs):
		return tf.nn.dropout(tf.nn.embedding_lookup(self.lookup, inputs), rate=0.2)


class JointEmbeddingModel:
	def __init__(self, config):
		self.data_dir = config.data_dir
		self.model_name = config.model_name
		self.methname_len = config.methname_len  # the max length of method name
		self.apiseq_len = config.apiseq_len
		self.tokens_len = config.tokens_len
		self.desc_len = config.desc_len
		self.conf = config
		self.embed_dims = config.embed_dims
		self.lstm_dims = config.lstm_dims
		self.hidden_dims = config.hidden_dims
		self.astpath_len = config.astpath_len
		self.astpath_num = config.path_num
		self.batch_size = config.batch_size

		self.margin = 0.05

		self.init_embed_weights_methodname = config.init_embed_weights_methodname
		self.init_embed_weights_tokens = config.init_embed_weights_tokens
		self.init_embed_weights_desc = config.init_embed_weights_desc

		self.methodname = Input(shape=(self.methname_len,), batch_size=self.batch_size, dtype='int32', name='methodname')
		self.apiseq = Input(shape=(self.apiseq_len,), batch_size=self.batch_size, dtype='int32', name='apiseq')
		# self.tokens = Input(shape=(self.tokens_len,), batch_size=self.batch_size, dtype='int32', name='tokens')
		self.desc_good = Input(shape=(self.desc_len,), batch_size=self.batch_size, dtype='int32', name='desc_good')
		self.desc_bad = Input(shape=(self.desc_len,), batch_size=self.batch_size, dtype='int32', name='desc_bad')
		self.astpath = Input(shape=(self.astpath_num, self.astpath_len), batch_size=self.batch_size, dtype='int32', name='astpaths')
		self.firstNode = Input(shape=(self.astpath_num, 3), batch_size=self.batch_size, dtype='int32', name='firstNodes')
		self.lastNode = Input(shape=(self.astpath_num, 3), batch_size=self.batch_size, dtype='int32', name='lastNodes')

		# create path to store model Info
		if not os.path.exists(self.data_dir + 'model/' + self.model_name):
			os.makedirs(self.data_dir + 'model/' + self.model_name)

	def build(self):

		# 1 -- CodeNN
		methodname = Input(shape=(self.methname_len,), batch_size=self.batch_size, dtype='int32', name='methodname')
		apiseq = Input(shape=(self.apiseq_len,), batch_size=self.batch_size, dtype='int32', name='apiseq')
		# tokens = Input(shape=(self.tokens_len,), dtype='int32', name='tokens')

		##
		astpath = Input(shape=(self.astpath_num, self.astpath_len,), batch_size=self.batch_size, dtype='int32', name='astpaths')
		firstNode = Input(shape=(self.astpath_num, 3,), batch_size=self.batch_size, dtype='int32', name='firstNodes')
		lastNode = Input(shape=(self.astpath_num, 3,), batch_size=self.batch_size, dtype='int32', name='lastNodes')

		astpaths = tf.reshape(astpath, shape=(self.batch_size*self.astpath_num, self.astpath_len))
		firstNodes = tf.reshape(firstNode, shape=(self.batch_size*self.astpath_num, 3))
		lastNodes = tf.reshape(lastNode, shape=(self.batch_size*self.astpath_num, 3))
		##

		# # 2 tokens
		# # embedding layer
		# init_emd_weights = np.load(
		# 	self.data_dir + self.init_embed_weights_tokens) if self.init_embed_weights_tokens is not None else None
		# init_emd_weights = init_emd_weights if init_emd_weights is None else [init_emd_weights]
		#
		# embedding_tokens = Embedding(
		# 	input_dim=self.vocab_size,
		# 	output_dim=self.embed_dims,
		# 	weights=init_emd_weights,
		# 	mask_zero=False,
		# 	name='embedding_tokens'
		# )
		#
		# tokens_embedding = embedding_tokens(tokens)
		# # max pooling
		# tokens_pool = maxpool(tokens_dropout)
		# activation = Activation('tanh', name='active_tokens')
		# tokens_repr = activation(tokens_pool)

		# 3 ast
		embedding = Embedding(
			input_dim=self.conf.ast_words,
			output_dim=self.embed_dims,
			weights=None,
			mask_zero=False,
			name='embedding_ast'
		)

		nodembedding = Embedding(
			input_dim=self.conf.tokens_words,
			output_dim=self.embed_dims,
			weights=None,
			mask_zero=False,
			name='embedding_Node'
		)

		dropout = Dropout(0.25, name='dropout_ast_embed')

		# forward rnn
		fw_rnn = LSTM(self.lstm_dims, name='lstm_ast_fw')

		# backward rnn
		bw_rnn = LSTM(self.lstm_dims, go_backwards=True, name='lstm_ast_bw')

		##
		ast_embed = embedding(astpaths)
		astpath_fw = dropout(fw_rnn(ast_embed))
		astpath_bw = dropout(bw_rnn(ast_embed))
		first_mask = tf.not_equal(firstNodes, 0)
		first_mask = tf.tile(tf.expand_dims(first_mask, axis=2), (1, 1, self.embed_dims))
		last_mask = tf.not_equal(lastNodes, 0)
		last_mask = tf.tile(tf.expand_dims(last_mask, axis=2), (1, 1, self.embed_dims))

		first_embed = nodembedding(firstNodes)
		last_embed = nodembedding(lastNodes)
		zeros = tf.zeros_like(first_embed)
		first_embed = tf.where(first_mask, first_embed, zeros)
		last_embed = tf.where(last_mask, last_embed, zeros)
		astpath_fw = tf.reshape(astpath_fw, shape=(-1, self.astpath_num, self.lstm_dims))
		astpath_bw = tf.reshape(astpath_bw, shape=(-1, self.astpath_num, self.lstm_dims))
		first_embed = tf.reshape(first_embed, shape=(-1, self.astpath_num, 3, self.embed_dims))
		last_embed = tf.reshape(last_embed, shape=(-1, self.astpath_num, 3, self.embed_dims))

		# maxpool = Lambda(lambda x: K.mean(x, axis=2, keepdims=False), output_shape=lambda x: (x[0], x[1], x[3]),
		#                  name='maxpooling_ast')
		sumpool = Lambda(lambda x: K.max(x, axis=2, keepdims=False), output_shape=lambda x: (x[0], x[1], x[3]),
		                 name='maxpooling_node')

		# astpath_fw_pool = maxpool(astpath_fw)
		# astpath_bw_pool = maxpool(astpath_bw)
		first_sumpool = sumpool(first_embed)
		last_sumpool = sumpool(last_embed)
		astpath_concat = Concatenate(axis=2, name='astpathConcat')([astpath_fw, astpath_bw, first_sumpool, last_sumpool])
		astpath_out = Dense(self.lstm_dims, 'relu', name='astpath')(astpath_concat)

		meanpool = Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='meanpooling_astpaths')
		astpath_repr = meanpool(astpath_out)
		##

		# 4 methodname
		# embedding layer
		init_emd_weights = np.load(
			self.data_dir + self.init_embed_weights_methodname) if self.init_embed_weights_methodname is not None else None
		init_emd_weights = init_emd_weights if init_emd_weights is None else [init_emd_weights]

		embedding = Embedding(
			input_dim=self.conf.methname_words,
			output_dim=self.embed_dims,
			weights=init_emd_weights,
			mask_zero=False,
			name='embedding_methodname'
		)

		methodname_embedding = embedding(methodname)

		# dropout
		dropout = Dropout(0.25, name='dropout_methodname_embed')
		methodname_dropout = dropout(methodname_embedding)

		# forward rnn
		fw_rnn = LSTM(self.lstm_dims, return_sequences=True, name='lstm_methodname_fw')

		# backward rnn
		bw_rnn = LSTM(self.lstm_dims, return_sequences=True, go_backwards=True, name='lstm_methodname_bw')

		methodname_fw = fw_rnn(methodname_dropout)
		methodname_bw = bw_rnn(methodname_dropout)

		dropout = Dropout(0.25, name='dropout_methodname_rnn')
		methodname_fw_dropout = dropout(methodname_fw)
		methodname_bw_dropout = dropout(methodname_bw)

		# max pooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_methodname')
		methodname_pool = Concatenate(name='concat_methodname_lstm')(
			[maxpool(methodname_fw_dropout), maxpool(methodname_bw_dropout)])
		activation = Activation('relu', name='active_methodname')
		methodname_repr = activation(methodname_pool)

		# 5 apiseq
		# embedding layer
		embedding = Embedding(
			input_dim=self.conf.api_words,
			output_dim=self.embed_dims,
			mask_zero=False,
			name='embedding_apiseq'
		)

		apiseq_embedding = embedding(apiseq)

		# dropout
		dropout = Dropout(0.25, name='dropout_apiseq_embed')
		apiseq_dropout = dropout(apiseq_embedding)

		# forward rnn
		fw_rnn = LSTM(self.lstm_dims, return_sequences=True, name='lstm_apiseq_fw')

		# backward rnn
		bw_rnn = LSTM(self.lstm_dims, return_sequences=True, go_backwards=True, name='lstm_apiseq_bw')

		apiseq_fw = fw_rnn(apiseq_dropout)
		apiseq_bw = bw_rnn(apiseq_dropout)

		dropout = Dropout(0.25, name='dropout_apiseq_rnn')
		apiseq_fw_dropout = dropout(apiseq_fw)
		apiseq_bw_dropout = dropout(apiseq_bw)

		# max pooling

		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_apiseq')
		apiseq_pool = Concatenate(name='concat_apiseq_lstm')([maxpool(apiseq_fw_dropout), maxpool(apiseq_bw_dropout)])
		activation = Activation('relu', name='active_apiseq')
		apiseq_repr = activation(apiseq_pool)


		# fusion methodname, apiseq, tokens
		merge_ast_repr = Concatenate(name='merge_methname_ast')([methodname_repr, astpath_repr, apiseq_repr])
		code_repr = merge_ast_repr

		code_repr = Dense(2*self.lstm_dims, name='dense_coderepr')(code_repr)

		self.code_repr_model = Model(inputs=[methodname, apiseq, astpath, firstNode, lastNode],
		                             outputs=[code_repr], name='code_repr_model')
		self.code_repr_model.summary()

		#  2 -- description
		desc = Input(shape=(self.desc_len,), batch_size=self.batch_size, dtype='int32', name='desc')

		# desc
		# embedding layer
		init_emd_weights = np.load(
			self.data_dir + self.init_embed_weights_desc) if self.init_embed_weights_desc is not None else None
		init_emd_weights = init_emd_weights if init_emd_weights is None else [init_emd_weights]

		embedding = Embedding(
			input_dim=self.conf.desc_words,
			output_dim=self.embed_dims,
			weights=init_emd_weights,
			mask_zero=False,
			name='embedding_desc'
		)

		desc_embedding = embedding(desc)

		# dropout
		dropout = Dropout(0.25, name='dropout_desc_embed')
		desc_dropout = dropout(desc_embedding)

		# forward rnn
		fw_rnn = LSTM(self.lstm_dims, return_sequences=True, name='lstm_desc_fw')

		# backward rnn
		bw_rnn = LSTM(self.lstm_dims, return_sequences=True, go_backwards=True, name='lstm_desc_bw')

		desc_fw = fw_rnn(desc_dropout)
		desc_bw = bw_rnn(desc_dropout)

		dropout = Dropout(0.25, name='dropout_desc_rnn')
		desc_fw_dropout = dropout(desc_fw)
		desc_bw_dropout = dropout(desc_bw)

		# max pooling

		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_desc')
		desc_pool = Concatenate(name='concat_desc_lstm')([maxpool(desc_fw_dropout), maxpool(desc_bw_dropout)])
		activation = Activation('relu', name='active_desc')
		desc_repr = activation(desc_pool)

		self.desc_repr_model = Model(inputs=[desc], outputs=[desc_repr], name='desc_repr_model')
		self.desc_repr_model.summary()

		#  3 -- cosine similarity
		##################
		code_repr_v = self.code_repr_model([self.methodname, self.apiseq, self.astpath, self.firstNode, self.lastNode])
		good_desc_repr = self.desc_repr_model([self.desc_good])
		bad_desc_repr = self.desc_repr_model([self.desc_bad])
		good_sim = Dot(axes=1, normalize=True, name='cos_sim_good')([code_repr_v, good_desc_repr])
		bad_sim = Dot(axes=1, normalize=True, name='cos_sim_bad')([code_repr_v, bad_desc_repr])
		loss = Lambda(lambda x: K.maximum(1e-9, self.margin - x[0] + x[1]), output_shape=lambda x: x[0], name='loss')(
			[good_sim, bad_sim])

		self.training_model = Model(
			inputs=[self.methodname, self.apiseq, self.astpath, self.firstNode, self.lastNode, self.desc_good, self.desc_bad],
			outputs=[loss], name='training_model')
		self.training_model.summary()

	def compile(self, **kwargs):
		optimizer = optimizers.Adam(lr=0.001)
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
		pre = Dot(axes=1, normalize=True, name="cos_sim")([self.repr_code(x[:5]), self.repr_desc(x[5])])
		return np.array(pre).flatten()

	def save(self, code_model_file, desc_model_file, **kwargs):
		self.code_repr_model.save_weights(code_model_file, **kwargs)
		self.desc_repr_model.save_weights(desc_model_file, **kwargs)

	def load(self, code_model_file, desc_model_file, **kwargs):
		self.code_repr_model.load_weights(code_model_file, **kwargs)
		self.desc_repr_model.load_weights(desc_model_file, **kwargs)
