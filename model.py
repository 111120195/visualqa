import keras.backend as K
from keras import Input
from keras.layers import GRU, Bidirectional, Dropout, Dense, Embedding, Lambda, concatenate, Reshape
from keras.models import Model
from config import Config
from data_generater import DataGenerate


def visual_question_feature_embedding(visual_feature, question_feature, embedding_dim=100):
	"""
	:param question_feature:
	:param visual_feature: output of snack_reshape
	:param embedding_size:
	:return:
	"""
	gru_hidding_size = 256
	q = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=query_maxlen)(
		question_feature)  # (samples, query_maxlen, embedding_dim)
	q = Dropout(0.5)(q)
	q_encoder = Bidirectional(GRU(gru_hidding_size), merge_mode='ave')(q)

	v = Dense(embedding_dim, activation='tanh')(visual_feature)
	v = Dropout(0.5)(v)
	v_encoder = Bidirectional(
		GRU(units=gru_hidding_size, return_sequences=True), merge_mode='ave')(v)
	return v_encoder, q_encoder


def gate(inputs):
	v_encoder, q_encoder = inputs
	z = K.concatenate([v_encoder * Reshape((1, 256))(q_encoder), v_encoder * Reshape((1, 256))(q_encoder),
					   K.abs(v_encoder - Reshape((1, 256))(q_encoder)),
					   K.abs(v_encoder - Reshape((1, 256))(q_encoder))], axis=-1)
	return z


def episodic_memory_module(visual_feature, question_feature, pre_memory):
	"""
	soft attention
	:param visual_feature:  shape: (batch_size, 14*14(local_size) , embedding_size)
	:param question_feature: shape: (batch_size, embedding_size)
	:param pre_memory: shape: (batch_size, embeding_size)
	:return:
	"""
	# calculate update gate z
	z = Lambda(gate)([visual_feature, question_feature])
	z = Dense(512, activation='tanh')(z)
	z = Dropout(0.5)(z)
	# softmax_size = visual_feature.shape[1]  # softmax_size = embdding_size
	z = Dense(1, activation='sigmoid')(z)  # shape: (batch_size, softmax_size)

	c = Lambda(lambda inputs: K.sum(inputs[0] * inputs[1], axis=1))([visual_feature, z])
	cur_memory = Dense(512, activation='relu')(concatenate([pre_memory, c, question_feature], axis=-1))
	cur_memory = Dropout(0.5)(cur_memory)
	return cur_memory


def answer_module(memory, question, answer_size):
	answer = Dense(answer_size, activation='sigmoid')(concatenate([memory, question]))
	return answer


def build_model(input_v, input_q, answer_size):
	"""
	build and complie model
	:param input_v: visual local feature with shape of (14*14, 512)
	:param input_q: question encoded with vector
	:return: model
	"""
	v_encoder, q_encoder = visual_question_feature_embedding(input_v, input_q)
	cur_memory = episodic_memory_module(v_encoder, q_encoder, q_encoder)
	cur_memory = episodic_memory_module(v_encoder, q_encoder, cur_memory)
	output = answer_module(cur_memory, input_q, answer_size)

	model = Model([input_v, input_q], output)

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model


if __name__ == '__main__':
	config = Config()
	data = DataGenerate(config)
	train = data.generate_data()

	setting = data.get_config()
	answer_size = setting['answer_word_size']
	steps_per_epoch = setting['steps_per_epoch']
	vocab_size = setting['vocab_size']
	query_maxlen = setting['max_question_size']

	input_v = Input(shape=(14 * 14, 512))
	input_q = Input(shape=(query_maxlen,))
	# data._encode_image()

	model = build_model(input_v, input_q, answer_size)
	print(model.summary())

	model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=5)

	model.save('model.h5')
