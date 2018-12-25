import keras.backend as K
from keras import Input
from keras.layers import GRU, Bidirectional, Dropout, Dense, Embedding, Lambda, concatenate, Reshape, \
	BatchNormalization, Permute, Softmax, Multiply
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop

from config import Config
from data_generater import DataGenerate


class VqaModel(object):
	def __init__(self, config, setting):
		self.config = config
		self.data_info = setting

	def visual_question_feature_embedding(self, visual_feature, question_feature):
		"""
		:param question_feature:
		:param visual_feature: output of snack_reshape
		"""
		config = self.config
		setting = self.data_info

		gru_hidden_size = config.gru_hidden_size
		embedding_dim = config.word_embedding_size
		drop_out_rate = config.drop_out_rate

		vocab_size = setting['vocab_size']
		query_maxlen = setting['max_question_size']

		q = Embedding(input_dim=vocab_size + 2,
					  output_dim=embedding_dim,
					  input_length=query_maxlen)(question_feature)
		# (samples, query_maxlen, embedding_dim)
		# TODO using Bert language model
		q = Dropout(drop_out_rate)(q)
		# q_encoder = Bidirectional(GRU(gru_hidden_size), merge_mode='ave')(q)
		# Bidirectional is meaningless because only last state output
		q_encoder = GRU(gru_hidden_size, dropout=drop_out_rate)(q)

		v = Dense(embedding_dim, activation='tanh')(visual_feature)  # meaning? necessary?
		v = Dropout(drop_out_rate)(v)
		v_encoder = Bidirectional(GRU(units=gru_hidden_size,
									  return_sequences=True,
									  dropout=drop_out_rate),
								  merge_mode='ave')(v)

		return v_encoder, q_encoder

	def gate(self, inputs):
		v_encoder, q_encoder, pre_memory = inputs

		z = K.concatenate([v_encoder * Reshape((1, self.config.gru_hidden_size))(q_encoder),
						   v_encoder * Reshape((1, self.config.gru_hidden_size))(pre_memory),
						   K.abs(v_encoder - Reshape((1, self.config.gru_hidden_size))(q_encoder)),
						   K.abs(v_encoder - Reshape((1, self.config.gru_hidden_size))(pre_memory))], axis=-1)
		return z

	@staticmethod
	def softmax_norm(inputs):
		return K.exp(inputs) / K.sum(K.exp(inputs))

	def episodic_memory_module(self, visual_feature, question_feature, pre_memory):
		"""
		soft attention
		:param visual_feature:  shape: (batch_size, 14*14(local_size) , embedding_size)
		:param question_feature: shape: (batch_size, embedding_size)
		:param pre_memory: shape: (batch_size, embeding_size)
		:return:
		"""
		# calculate update gate z
		z = Lambda(self.gate)([visual_feature, question_feature, pre_memory])
		z = Dense(self.config.episodic_memory_hidden_size, activation='tanh')(z)
		z = Dropout(self.config.drop_out_rate)(z)
		z = Dense(1)(z)  # shape: (batch_size, 196, 1)
		# z = Lambda(self.softmax_norm)(z)
		# z = Permute(dims=(0, 2, 1))(z)
		z = Lambda(K.softmax)(z)
		# z = Softmax(axis=1)(z)
		c = Lambda(lambda inputs: K.sum(inputs[0] * inputs[1], axis=1))([visual_feature, z])
		# TODO using GRU attention

		cur_memory = Dense(self.config.memory_size, activation='relu')(
			concatenate([pre_memory, c, question_feature], axis=-1))
		cur_memory = Dropout(self.config.drop_out_rate)(cur_memory)

		return cur_memory

	def answer_module(self, memory, question):
		answer_classes = self.data_info['answer_word_size']

		answer = Dense(self.config.answer_output_hidden_size,
					   activation='relu')(concatenate([memory, question]))
		answer = Dropout(self.config.drop_out_rate)(answer)

		if answer_classes == 2:
			answer = Dense(1, activation='sigmoid')(answer)
		else:
			answer = Dense(answer_classes + 1, activation='softmax')(answer)

		return answer

	def build_model(self):
		"""
		build and compile model
		"""
		query_maxlen = self.data_info['max_question_size']
		v_input = Input(shape=self.config.image_input_shape)
		v_input_norm = Lambda(K.l2_normalize)(v_input)

		q_input = Input(shape=(query_maxlen,))
		v_encoder, q_encoder = self.visual_question_feature_embedding(v_input_norm, q_input)

		cur_memory = self.episodic_memory_module(v_encoder, q_encoder, q_encoder)
		cur_memory = self.episodic_memory_module(v_encoder, q_encoder, cur_memory)
		output = self.answer_module(cur_memory, q_encoder)
		# TODO modify net structure
		_model = Model([v_input, q_input], output)

		if self.config.optimizer == 'sgd':
			opt = SGD(lr=self.config.lr)
		elif self.config.optimizer == 'rmsprop':
			opt = RMSprop(lr=self.config.lr)
		else:
			opt = Adam(lr=self.config.lr)
		answer_classes = self.data_info['answer_word_size']

		if answer_classes == 2:
			_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
		else:
			_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return _model


if __name__ == '__main__':
	config_ = Config()
	data = DataGenerate(config_)
	train = data.generate_data(baseline=False, data_type='train')
	val = data.generate_data(baseline=False, data_type='val')

	setting = data.get_data_info()
	answer_size = setting['answer_word_size']
	steps_per_epoch = setting['steps_per_epoch']
	validation_steps = setting['validation_steps']
	vocab_size = setting['vocab_size']
	query_maxlen = setting['max_question_size']
	#

	model = VqaModel(config_, setting).build_model()
	print(model.summary())
# TODO select fit parameter
# model.load_weights('model.h5')
#
# checkpoint = ModelCheckpoint('vqa.{epoch:02d-{val_loss:.2f}}.h5', monitor='val_loss', verbose=1,
# 							 save_best_only=True,
# 							 mode='min', period=1)
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#
# model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=100, validation_data=val,
# 					validation_steps=validation_steps, callbacks=[checkpoint, early_stopping])
#
# model.save('vqa.h5')
