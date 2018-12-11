from keras.layers import GRU, Dense, Dropout, Bidirectional, Embedding, Dot, Reshape
from keras.layers import Input
from keras.layers.merge import concatenate, subtract, multiply, add
from keras.layers.core import RepeatVector, Lambda
from keras.models import Model
import keras.backend as K


def episodic_memory_module(visual_feature, question_feature, pre_memory):
	"""
	soft attention
	:param visual_feature:  shape: (batch_size, 14*14(local_size) , embedding_size)
	:param question_feature: shape: (batch_size, embedding_size)
	:param pre_memory: shape: (batch_size, embeding_size)
	:return:
	"""
	# calculate update gate z
	z = K.concatenate([visual_feature * question_feature, visual_feature * pre_memory,
					   K.abs(visual_feature - question_feature),
					   K.abs(visual_feature - pre_memory)], axis=-1)
	z = Dense(512, activation='tanh')(z)
	# softmax_size = visual_feature.shape[1]  # softmax_size = embdding_size
	z = Dense(512, activation='softmax')(z)  # shape: (batch_size, softmax_size)

	c = K.sum(Dot(visual_feature, z, axes=1), axis=1)  # shape: (batch_size, softmax_size)
	cur_memory = Dense(512, activation='relu')(K.concatenate([pre_memory, c, question_feature], axis=-1))
	return cur_memory


if __name__ == '__main__':
	input_v = Input(shape=(14 * 14, 512))
	input_q = Input(shape=(23,))

	vocab_size = 10000
	query_maxlen = 23
	gru_hidding_size = 256
	embedding_dim = 512
	q = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=query_maxlen)(
		input_q)  # (samples, query_maxlen, embedding_dim)
	q = Dropout(0.5)(q)
	q_encoder = Bidirectional(GRU(gru_hidding_size), merge_mode='ave')(q)  # (samples, 256)

	v = Dense(embedding_dim, activation='tanh')(input_v)
	v = Dropout(0.5)(v)
	v_encoder = Bidirectional(
		GRU(units=gru_hidding_size, return_sequences=True), merge_mode='ave')(v)  # (samples, 196, 256)


	# memory = episodic_memory_module(visual_feature=v_encoder, question_feature=q_encoder, pre_memory=q_encoder)
	def foo(inputs):
		v_encoder, q_encoder = inputs
		z = K.concatenate([v_encoder * q_encoder, v_encoder * q_encoder,
						   K.abs(v_encoder - Reshape((1, 256))(q_encoder)),
						   K.abs(v_encoder - Reshape((1, 256))(q_encoder))], axis=-1)
		return z


	z = Lambda(foo)([v_encoder, q_encoder])

	print(z)
	z = Dense(512, activation='tanh')(z)
	print(z)
	z = Dense(1, activation='sigmoid')(z)  # shape: (batch_size, softmax_size)
	print(z)

	# c = K.sum(d, axis=1)  # shape: (batc = Lambda(lambda inputs: K.sum(inputs[0] * inputs[1], axis=1))([v_encoder, z])ch_size, softmax_size)
	print('c:', c)
	m = concatenate([q_encoder, c, q_encoder], axis=-1)
	print('m:', m)
	cur_memory = Dense(512, activation='relu')(m)
	print('cur_memory:', cur_memory)
# model = Model([input_q, input_v], z)
# # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
