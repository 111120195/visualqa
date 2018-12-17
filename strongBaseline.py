import keras.backend as K
from keras import Model
from keras.applications import ResNet50
from keras.layers import Input, Dropout, Dense, Conv2D, LSTM, Embedding, RepeatVector, Reshape, concatenate, \
	Lambda
from keras.layers.merge import multiply


def build_model():
	pre_trained = ResNet50(include_top=False,
						   weights='imagenet',
						   input_shape=(224, 224, 3))
	print(pre_trained.summary())


if __name__ == '__main__':
	pre_trained = ResNet50(include_top=False,
						   weights='imagenet',
						   input_shape=(224, 224, 3))

	for layer in pre_trained.layers:
		layer.trainable = False

	image_input = Input(shape=(224, 224, 3))
	image_feature = pre_trained(image_input)
	image_feature = Lambda(lambda x: K.l2_normalize(x, axis=-1))(image_feature)

	question_input = Input(shape=(15,))
	q_embedding = Embedding(input_dim=6000, input_length=15, output_dim=300)(question_input)
	q_embedding = Dropout(0.5)(q_embedding)
	q_encoder = LSTM(1024, dropout=0.5)(q_embedding)

	q_encoder_ = RepeatVector(49)(q_encoder)
	q_encoder_ = Reshape(input_shape=(49, 1024), target_shape=(7, 7, 1024))(q_encoder_)
	q_v_concat = concatenate([image_feature, q_encoder_], axis=-1)

	v_weight = Conv2D(filters=512, kernel_size=1, activation='relu')(q_v_concat)
	v_weight1 = Conv2D(filters=1, kernel_size=1, activation='softmax')(v_weight)
	v_weight2 = Conv2D(filters=1, kernel_size=1, activation='softmax')(v_weight)

	glimpse1 = multiply(inputs=[image_feature, v_weight1])
	glimpse1 = Reshape(target_shape=(49, 2048))(glimpse1)
	glimpse1 = Lambda(lambda inputs: K.sum(inputs, axis=1))(glimpse1)

	glimpse2 = multiply(inputs=[image_feature, v_weight2])
	glimpse2 = Reshape(target_shape=(49, 2048))(glimpse2)
	glimpse2 = Lambda(lambda inputs: K.sum(inputs, axis=1))(glimpse2)

	answer_input = concatenate(inputs=[glimpse1, glimpse2, q_encoder], axis=-1)
	answer = Dense(1024, activation='relu')(answer_input)
	answer = Dropout(0.5)(answer)
	answer = Dense(3000, activation='softmax')(answer)

	model = Model([image_input, question_input], answer)

	print(model.summary())
