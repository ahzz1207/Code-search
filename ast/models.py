import numpy as np
from tensorflow.python.keras.layers import Concatenate, Dot, Embedding, Lambda, Activation, Dense, Dropout, add
from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.engine import Input
from tensorflow.python.keras import backend as K
from tensorflow.python import keras
from tensorflow.python.keras import optimizers
import tensorflow as tf
import os


class JointEmbeddingModel:
	def __init__(self, config):
		self.data_dir = config.data_dir
		self.model_name = config.model_name
		self.methname_len = config.methname_len  # the max length of method name
		self.apiseq_len = config.apiseq_len
		self.tokens_len = config.tokens_len
		self.desc_len = config.desc_len
		self.vocab_size = config.n_words  # the size of vocab
		self.embed_dims = config.embed_dims
		self.lstm_dims = config.lstm_dims
		self.hidden_dims = config.hidden_dims


		self.margin = 0.05

		self.init_embed_weights_methodname = config.init_embed_weights_methodname
		self.init_embed_weights_tokens = config.init_embed_weights_tokens
		self.init_embed_weights_desc = config.init_embed_weights_desc

		self.methodname = Input(shape=(self.methname_len,), dtype='int32', name='methodname')
		self.apiseq = Input(shape=(self.apiseq_len,), dtype='int32', name='apiseq')
		self.tokens = Input(shape=(self.tokens_len,), dtype='int32', name='tokens')
		self.desc_good = Input(shape=(self.desc_len,), dtype='int32', name='desc_good')
		self.desc_bad = Input(shape=(self.desc_len,), dtype='int32', name='desc_bad')
		self.astpath = []
		for i in range(self.astpath_num):
			self.astpath.append(Input(shape=(self.astpath_len,), dtype='int32', name='astpath' + str(i)))

		# create path to store model Info
		if not os.path.exists(self.data_dir + 'model/' + self.model_name):
			os.makedirs(self.data_dir + 'model/' + self.model_name)

	def build(self):

		# 1 -- CodeNN
		methodname = Input(shape=(self.methname_len,), dtype='int32', name='methodname')
		apiseq = Input(shape=(self.apiseq_len,), dtype='int32', name='apiseq')
		tokens = Input(shape=(self.tokens_len,), dtype='int32', name='tokens')
		astpath = []
		leafNode = []
		for i in range(self.astpath_num):
			astpath.append(Input(shape=(self.astpath_len,), dtype='int32', name='astpath' + str(i)))
			leafNode.append(Input(shape=(5,), dtype='int32', name='leaf' + str(i)))

		# ast
		embedding = Embedding(
			input_dim=self.ast_words,
			output_dim=self.embed_dims,
			weights=None,
			mask_zero=False,
			name='embedding_ast'
		)
		dropout = Dropout(0.25, name='dropout_ast_embed')

		# forward rnn
		fw_rnn = LSTM(self.lstm_dims, return_sequences=True, name='lstm_ast_fw')

		# backward rnn
		bw_rnn = LSTM(self.lstm_dims, return_sequences=True, go_backwards=True, name='lstm_ast_bw')

		astpath_embed = [embedding(x) for x in astpath]
		# astpath_embed = dropout(astpath_embed)
		# astpath_out_fw = dropout(add([fw_rnn(x) for x in astpath_embed]))
		# astpath_out_bw = dropout(add([bw_rnn(x) for x in astpath_embed]))
		astpath_out_fw = fw_rnn(astpath_embed[0])
		for i in range(1, self.astpath_num):
			astpath_out_fw = add([astpath_out_fw, fw_rnn(astpath_embed[i])])
		# astpath_out_fw = dropout(astpath_out_fw)

		astpath_out_bw = bw_rnn(astpath_embed[0])
		for i in range(1, self.astpath_num):
			astpath_out_bw = add([astpath_out_bw, bw_rnn(astpath_embed[i])])
		# astpath_out_bw = dropout(astpath_out_bw)

		# todo:average pooling
		avepool = Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                     name='averagepooling_astpath')
		# astpath_pool = maxpool(astpath_out_fw)
		astpath_pool = Concatenate(name='ast_concatenate', )([avepool(astpath_out_fw), avepool(astpath_out_bw)])
		# fully connection
		astpath_fully_repr = Dense(self.hidden_dims, 'tanh', name='fully_connect_astpath')(astpath_pool)

		# methodname
		# embedding layer
		init_emd_weights = np.load(
			self.data_dir + self.init_embed_weights_methodname) if self.init_embed_weights_methodname is not None else None
		init_emd_weights = init_emd_weights if init_emd_weights is None else [init_emd_weights]

		embedding = Embedding(
			input_dim=self.vocab_size,
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
		activation = Activation('tanh', name='active_methodname')
		methodname_repr = activation(methodname_pool)

		# apiseq
		# embedding layer
		embedding = Embedding(
			input_dim=self.vocab_size,
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

		# apiseq_fw = fw_rnn(apiseq_dropout)
		# apiseq_bw = bw_rnn(apiseq_dropout)
		#
		# dropout = Dropout(0.25, name='dropout_apiseq_rnn')
		# apiseq_fw_dropout = dropout(apiseq_fw)
		# apiseq_bw_dropout = dropout(apiseq_bw)

		# max pooling

		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_apiseq')
		# apiseq_pool = Concatenate(name='concat_apiseq_lstm')([maxpool(apiseq_fw_dropout), maxpool(apiseq_bw_dropout)])
		apiseq_pool = maxpool(apiseq_dropout)
		activation = Activation('tanh', name='active_apiseq')
		apiseq_repr = activation(apiseq_pool)

		# tokens
		# embedding layer
		init_emd_weights = np.load(
			self.data_dir + self.init_embed_weights_tokens) if self.init_embed_weights_tokens is not None else None
		init_emd_weights = init_emd_weights if init_emd_weights is None else [init_emd_weights]

		embedding = Embedding(
			input_dim=self.vocab_size,
			output_dim=self.embed_dims,
			weights=init_emd_weights,
			mask_zero=False,
			name='embedding_tokens'
		)

		tokens_embedding = embedding(tokens)

		# dropout
		dropout = Dropout(0.25, name='dropout_tokens_embed')
		tokens_dropout = dropout(tokens_embedding)

		# # forward rnn
		# fw_rnn = LSTM(self.lstm_dims, return_sequences=True, name='lstm_tokens_fw')
		#
		# # backward rnn
		# bw_rnn = LSTM(self.lstm_dims, return_sequences=True, go_backwards=True, name='lstm_tokens_bw')
		#
		# tokens_fw = fw_rnn(tokens_dropout)
		# tokens_bw = bw_rnn(tokens_dropout)
		#
		# dropout = Dropout(0.25, name='dropout_tokens_rnn')
		# tokens_fw_dropout = dropout(tokens_fw)
		# tokens_bw_dropout = dropout(tokens_bw)

		# max pooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_tokens')
		# tokens_pool = Concatenate(name='concat_tokens_lstm')([maxpool(tokens_fw_dropout), maxpool(tokens_bw_dropout)])
		tokens_pool = maxpool(tokens_dropout)
		activation = Activation('tanh', name='active_tokens')
		tokens_repr = activation(tokens_pool)

		# fusion methodname, apiseq, tokens
		merge_ast_repr = Concatenate(name='merge_methname_ast')([methodname_repr, astpath_fully_repr])
		merge_methname_api = Concatenate(name='merge_methname_api')([merge_ast_repr, apiseq_repr])
		merge_code_repr = Concatenate(name='merge_code_repr')([merge_methname_api, tokens_repr])

		code_repr = Dense(self.hidden_dims, activation='tanh', name='dense_coderepr')(merge_code_repr)

		self.code_repr_model = Model(inputs=[methodname, apiseq, tokens, astpath], outputs=[code_repr], name='code_repr_model')
		self.code_repr_model.summary()

		#  2 -- description
		desc = Input(shape=(self.desc_len,), dtype='int32', name='desc')

		# desc
		# embedding layer
		init_emd_weights = np.load(
			self.data_dir + self.init_embed_weights_desc) if self.init_embed_weights_desc is not None else None
		init_emd_weights = init_emd_weights if init_emd_weights is None else [init_emd_weights]

		embedding = Embedding(
			input_dim=self.vocab_size,
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
		activation = Activation('tanh', name='active_desc')
		desc_repr = activation(desc_pool)

		self.desc_repr_model = Model(inputs=[desc], outputs=[desc_repr], name='desc_repr_model')
		self.desc_repr_model.summary()

		#  3 -- cosine similarity
		code_repr = self.code_repr_model([methodname, apiseq, tokens, astpath])
		desc_repr = self.desc_repr_model([desc])

		cos_sim = Dot(axes=1, normalize=True, name='cos_sim')([code_repr, desc_repr])

		sim_model = Model(inputs=[methodname, apiseq, tokens, astpath, desc], outputs=[cos_sim], name='sim_model')

		self.sim_model = sim_model

		self.sim_model.summary()

		#  4 -- build training model
		good_sim = sim_model([self.methodname, self.apiseq, self.tokens, self.astpath, self.desc_good])
		bad_sim = sim_model([self.methodname, self.apiseq, self.tokens, self.astpath, self.desc_bad])

		loss = Lambda(lambda x: K.maximum(1e-6, self.margin - x[0] + x[1]), output_shape=lambda x: x[0], name='loss')(
			[good_sim, bad_sim])

		self.training_model = Model(inputs=[self.methodname, self.apiseq, self.tokens, self.astpath, self.desc_good, self.desc_bad],
		                            outputs=[loss], name='training_model')

		self.training_model.summary()

	def compile(self, **kwargs):
		optimizer = optimizers.Adam(lr=0.001)
		self.code_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
		self.desc_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
		self.training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=optimizer, **kwargs)
		self.sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

	def fit(self, x, **kwargs):
		y = np.zeros(shape=x[0].shape[:1], dtype=np.float32)
		return self.training_model.fit(x, y, **kwargs)

	def repr_code(self, x, **kwargs):
		return self.code_repr_model.predict(x, **kwargs)

	def repr_desc(self, x, **kwargs):
		return self.desc_repr_model.predict(x, **kwargs)

	def predict(self, x, **kwargs):
		return self.sim_model.predict(x, **kwargs)

	def save(self, code_model_file, desc_model_file, **kwargs):
		self.code_repr_model.save_weights(code_model_file, **kwargs)
		self.desc_repr_model.save_weights(desc_model_file, **kwargs)

	def load(self, code_model_file, desc_model_file, **kwargs):
		self.code_repr_model.load_weights(code_model_file, **kwargs)
		self.desc_repr_model.load_weights(desc_model_file, **kwargs)
