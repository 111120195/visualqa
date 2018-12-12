import keras.backend as K
from keras import Input, optimizers
from keras.applications import VGG19
from keras.layers import GRU, Dropout, Dense, Embedding, concatenate
from keras.models import Model

from config import Config
from data_generater import DataGenerate


def visual_question_feature_embedding(question_feature):
	"""
	:param question_feature:
	:param visual_feature: output of snack_reshape
	:param embedding_size:
	:return:
	"""
	gru_hidding_size = 1000
	embedding_dim = 512
	q = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=query_maxlen)(
		question_feature)  # (samples, query_maxlen, embedding_dim)
	q = Dropout(0.5)(q)
	q_encoder = GRU(gru_hidding_size)(q)

	return q_encoder


def answer_module(virsual, question):
	answer = Dense(512, activation='relu')(concatenate([virsual, question]))
	answer = Dense(1, activation='sigmoid')(answer)
	return answer


def build_model():
	"""
	build and complie model
	:param input_v: visual local feature with shape of (14*14, 512)
	:param input_q: question encoded with vector
	:return: model
	"""
	vgg19 = VGG19(weights='imagenet', include_top=True)
	input_q = Input(shape=(query_maxlen,))

	layer_name = []
	for layer in vgg19.layers:
		layer.trainable = False
		layer_name.append(layer.name)

	q_encoder = visual_question_feature_embedding(input_q)
	x = vgg19.get_layer(layer_name[-2]).output
	x = Dense(1000, activation='relu')(x)
	output = answer_module(x, q_encoder)
	model = Model([vgg19.input, input_q], output)

	opt = optimizers.Adam(lr=0.001)

	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

	return model


if __name__ == '__main__':
	K.clear_session()
	config = Config()
	data = DataGenerate(config)
	train = data.generate_data(baseline=True)

	setting = data.get_config()
	answer_size = setting['answer_word_size'] - 1
	steps_per_epoch = setting['steps_per_epoch']
	vocab_size = setting['vocab_size']
	query_maxlen = setting['max_question_size']

	# data._encode_image()

	model = build_model()
	print(model.summary())

	model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=2)

	model.save('model.h5')
