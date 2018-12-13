import keras.backend as K
import numpy as np
from keras.layers import Dense, Dropout, Dot, BatchNormalization
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model


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
	video_c3d_input = Input(shape=(20, 4096), dtype=np.float32)
	video_xcep_input = Input(shape=(20, 1920), dtype=np.float32)
	video_question_input = Input(shape=(20,), dtype='int32')

	encoded_video_c3d = model_attention_applied_after_lstm(video_c3d_input, 20, 512, '1')
	encoded_video_xcep = model_attention_applied_after_lstm(video_xcep_input, 20, 1024, '2')

	encoded_video_question = encoded_video_question_create(video_question_input)
	merged = concatenate([
		encoded_video_c3d,
		encoded_video_xcep, encoded_video_question])

	merged = BatchNormalization()(merged)
	merged = Dense(1024)(merged)
	merged = Dropout(0.5)(merged)
	output = Dense(100, activation='softmax')(merged)

	vqa_model = Model(inputs=[
		video_c3d_input,
		video_xcep_input, video_question_input], outputs=output)

	# compile model
	vqa_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	vqa_model.summary()
