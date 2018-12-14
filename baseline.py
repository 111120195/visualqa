import keras.backend as K
from keras import Input, optimizers
from keras.applications import VGG19
from keras.layers import Dropout, Dense, Embedding, concatenate, Flatten, GRU
from keras.models import Model
from keras.utils import plot_model

from config import Config
from data_generater import DataGenerate


def build_model():
	"""
	build and compile model
	:param input_v: visual local feature with shape of (14*14, 512)
	:param input_q: question encoded with vector
	:return: model
	"""
	input_q = Input(shape=(query_maxlen,))
	input_x = Input(shape=(4096,))

	q = Embedding(input_dim=vocab_size + 2, output_dim=256, input_length=query_maxlen)(
		input_q)  # (samples, query_maxlen, embedding_dim)
	q = Dropout(0.5)(q)
	q = GRU(512)(q)

	merge_layer = concatenate([input_x, q])

	answer = Dense(512, activation='relu')(merge_layer)
	answer = Dropout(0.5)(answer)
	answer = Dense(1, activation='sigmoid')(answer)
	model = Model([input_x, input_q], answer)

	optimizer = optimizers.Adam(lr=0.0001)

	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

	return model


if __name__ == '__main__':
	K.clear_session()
	config = Config()
	data = DataGenerate(config)
	train = data.generate_data(baseline=True)

	setting = data.get_data_info()
	answer_size = setting['answer_word_size'] - 1
	steps_per_epoch = setting['steps_per_epoch']
	vocab_size = setting['vocab_size']
	query_maxlen = setting['max_question_size']

	# data._encode_image()

	base_model = build_model()
	print(base_model.summary())
	plot_model(base_model)

#
# base_model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=20)
#
# base_model.save('base_model.h5')
