import json
import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd

from keras.preprocessing.text import text_to_word_sequence
from config import Config


class DataParser(object):
	"""
	parse image_id,question and answer.
	generate DataFrame data with columns（image_id, question_code, answer_code）.
	generate vocab_dict.
	"""

	def __init__(self, _config):
		self.train_data = None
		self.val_data = None
		self.words2index = {}
		self.index2words = {}
		self.max_question_size = 0
		self.word_size = 0
		self.answer_word_size = 0
		self.max_answer_size = 0
		self._question_ids = set()
		self.train_sample_size = 0
		self.val_sample_size = 0
		self.config = _config
		self.answer_type_set = set()
		self.question_type_set = set()

	def parser_text_to_vocab(self, text):
		"""
		encode word list to index
		:param text: text list
		:return: index2word, word2index
		"""
		print('word size:%d' % self.word_size)
		# 0 is reserved for empty word, word_size+1 denote unknown word
		index2word = dict(zip(range(1, self.word_size + 1), text))
		word2index = dict(zip(text, range(1, self.word_size + 1)))
		return index2word, word2index

	def parse_answer(self, ann=None, answers_ls=None):
		"""
		parse answer json file to answer word list
		:param ann: answer json file
		:param answers_ls: answer word list
		:return: None
		"""
		for anns in ann['annotations']:
			# filter record not in config answer type if answer type exist
			if self.config.answer_type is not None and anns['answer_type'] not in self.config.answer_type:
				continue
			# filter record not in config question type if question type exist
			if self.config.question_type is not None and anns['question_type'] not in self.config.question_type:
				continue
			if self.config.answer_encode_format == 'softmax':
				#  filter answer more than single word.
				one_word_answer = [ax['answer'] for ax in anns['answers'] if
								   len(text_to_word_sequence(ax['answer'])) == 1]
				if not one_word_answer:
					continue
				counter = Counter(one_word_answer)
				[(w, v)] = counter.most_common(1)  # select most voted answer
				answers_ls.append([anns['image_id'], anns['question_id'], w])
				# add answer type and question type to set
				print(w)
				self.answer_type_set.add(anns['answer_type'])
				self.question_type_set.add(anns['question_type'])
			else:
				# parser answer to word list
				[(w, v)] = Counter([ax['answer'] for ax in anns['answers']]).most_common(1)  # select most voted answer
				answers_ls.append(
					[anns['image_id'], anns['question_id'], w])
			# used for question filter
			self._question_ids.add(anns['question_id'])

	def parse_question(self, ques=None, questions_ls=None):
		"""
		parser question json file to question word list
		:param ques: question json file
		:param questions_ls: question list
		:return: None
		"""
		for que in ques['questions']:
			# filter question
			if que['question_id'] not in self._question_ids:
				continue
			ques_ls = text_to_word_sequence(que['question'])
			if len(ques_ls) > self.max_question_size:
				self.max_question_size = len(ques_ls)  # compute question max length
			questions_ls.append([que['image_id'], que['question_id'], que['question']])

	def build_vocab(self, questions, answers):
		"""
		build vocab dict with given question and answer list
		:param questions: question word set
		:param answers: answer word set
		:return: None
		"""
		if self.config.answer_encode_format != 'softmax':
			# answer is sentence set
			answers = ' '.join(answers)
			answers = set([x.strip() for x in re.split('([\d\W])', answers) if x.strip()])

		question_sub_answer = questions - answers
		# answer word is ahead on question word in oder to encode answer only with word appear on answers
		word_list = sorted(list(answers)) + sorted(list(question_sub_answer))

		self.word_size = len(word_list)
		self.answer_word_size = len(answers)
		self.index2words, self.words2index = self.parser_text_to_vocab(word_list)
		print('complete create question and answer word set')

	def encode_answer(self, answer):
		"""
		encode answer to index
		:param answer: answer word or answer sentence
		:return: int or list of int
		"""
		# assert len(answers) == 10, 'the answers don\'t has 10'
		if self.config.answer_encode_format == 'softmax':
			index = self.words2index[answer] if answer in self.words2index else 0
			return index if index <= 3000 else 0
		else:
			return [self.words2index[x.strip()] for x in re.split('([\d\W])', answer) if x.strip()]

	def decode_answer(self, answer_encoder, size=1):
		"""
		decode answer index to word string
		only used for answer softmax encoder
		:param size: result number
		:param answer_encoder:
		:return: string:answer result
		"""
		sorted_index = np.argsort(answer_encoder)[:size]
		answer_res = []
		for index in sorted_index:
			word = self.index2words[index]
			prop = answer_encoder[index]
			answer_res.append(word + ':' + str(prop))
		return '\n'.join(answer_res)

	def encode_question(self, question):
		"""
		encode question sentence to index list
		:param question: question sentence
		:return: question encoder
		"""
		# map the word to index. if the word is not exit in vocab dict, word_size+1 will encode the word
		encoder = [self.words2index[w] if w in self.words2index else self.word_size + 1 for w in
				   text_to_word_sequence(question)]
		# make question encoder have same length. 0 denote empty word.
		question_encoder = [0 for i in range(self.max_question_size - len(encoder))]
		question_encoder.extend(encoder)
		return question_encoder

	def parse(self):
		"""
		parse json file to generate table and vocab dict
		"""
		print('start loading json file...')
		train_ann = json.load(open(self.config.train_annFile, 'r'))
		train_ques = json.load(open(self.config.train_questionFile, 'r'))
		val_ann = json.load(open(self.config.val_annFile, 'r'))
		val_ques = json.load(open(self.config.val_questionFile, 'r'))
		print('load completed!')

		questions_train_ls = []
		questions_val_ls = []
		answers_train_ls = []
		answers_val_ls = []

		self.parse_answer(train_ann, answers_train_ls)
		self.parse_answer(val_ann, answers_val_ls)
		print('complete parser train data')
		self.parse_question(train_ques, questions_train_ls)
		self.parse_question(val_ques, questions_val_ls)
		print('complete parser val data')

		assert len(questions_train_ls) == len(answers_train_ls)
		assert len(questions_val_ls) == len(answers_val_ls)
		# check the data integrity

		questions = ' '.join([x[2] for x in questions_train_ls])
		questions = questions + ' ' + ' '.join([x[2] for x in questions_val_ls])
		q_counter = Counter(text_to_word_sequence(questions))

		questions = set(dict(q_counter.most_common(6000)).keys())
		# questions = set()
		an_ls = [x[2] for x in answers_train_ls] + [x[2] for x in answers_val_ls]
		a_counter = Counter(an_ls)
		answers = set(dict(a_counter.most_common(3000)).keys())

		self.build_vocab(questions, answers)
		print('complete build vocab')

		question_id_answer_train_df = pd.DataFrame(data=answers_train_ls, columns=['image_id', 'question_id', 'answer'])
		question_id_answer_val_df = pd.DataFrame(data=answers_val_ls, columns=['image_id', 'question_id', 'answer'])
		question_id_question_train_df = pd.DataFrame(data=questions_train_ls,
													 columns=['image_id', 'question_id', 'question'])
		question_id_question_val_df = pd.DataFrame(data=questions_val_ls,
												   columns=['image_id', 'question_id', 'question'])

		question_id_answer_train_df['answer'] = question_id_answer_train_df['answer'].apply(self.encode_answer)
		question_id_answer_val_df['answer'] = question_id_answer_val_df['answer'].apply(self.encode_answer)

		question_id_question_train_df['question'] = question_id_question_train_df['question'].apply(
			self.encode_question)
		question_id_question_val_df['question'] = question_id_question_val_df['question'].apply(self.encode_question)

		self.train_data = pd.merge(question_id_question_train_df, question_id_answer_train_df,
								   on=['image_id', 'question_id']).drop(['question_id'], axis=1)
		self.val_data = pd.merge(question_id_question_val_df, question_id_answer_val_df,
								 on=['image_id', 'question_id']).drop(['question_id'], axis=1)
		self.train_sample_size = len(question_id_answer_train_df)
		self.val_sample_size = len(question_id_answer_val_df)
		print('train_sample_size:%d\n val_sample_size:%d ' % (self.train_sample_size, self.val_sample_size))
		self.data_cleaning('train')
		self.data_cleaning('val')
		print('remove data which can not find picture')

	def info(self):
		"""
		print info of data
		"""
		info = ''
		info += "word size:%s\n" % self.word_size
		info += "answer size(classes):%s\n" % self.answer_word_size
		info += "max question size:%s\n" % self.max_question_size
		info += "train data size: %s\n" % self.train_sample_size
		info += "val data size: %s\n" % self.val_sample_size
		info += 'question_type:\n' + str(self.question_type_set) + '\n'
		info += 'answer_type:\n' + str(self.answer_type_set)
		print(info)

	def data_cleaning(self, data_type):
		"""
		cleaning data which can't find image file
		:return:
		"""
		cleaning_items = 0
		if data_type == 'train':
			data_ = self.train_data
			image_dir = self.config.train_img_dir
		else:
			data_ = self.val_data
			image_dir = self.config.val_img_dir

		for index in data_.index:
			image_file = image_dir + str(data_['image_id'][index]).zfill(12) + '.jpg'
			if not os.path.isfile(image_file):
				data_.drop(index, inplace=True)
				cleaning_items += 1
				if data_type == 'train':
					self.train_sample_size -= 1
				else:
					self.val_sample_size -= 1

		return cleaning_items

	def save_result(self):
		"""
		save train data and val data to csv file
		"""
		self.train_data.to_csv(self.config.train_data_file)
		self.val_data.to_csv(self.config.val_data_file)

	def save_data(self):
		"""
		save self object
		"""
		with open(self.config.save_data_file, 'bw') as f:
			pickle.dump(self, f)


if __name__ == '__main__':
	config = Config()
	data = DataParser(config)
	data.parse()
# data.save_result()
# data.info()
# generate_test = data.generate_data(100, data='val')
# generate_train = data.generate_data(100, data='train')
# [a, b], c = next(generate_train)
