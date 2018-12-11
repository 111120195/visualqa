import json
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd

from keras.preprocessing.text import text_to_word_sequence
from config import Config


class DataParser(object):

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
        encode text
        :param text: text list
        :return: index2word, word2index
        """
        print('word size:%d' % self.word_size)
        index2word = dict(zip(range(1, self.word_size + 1), text))
        word2index = dict(zip(text, range(1, self.word_size + 1)))
        return index2word, word2index

    def parse_answer(self, ann=None, answers_ls=None):
        for anns in ann['annotations']:
            if self.config.answer_type is not None and anns['answer_type'] not in self.config.answer_type:
                continue
            if self.config.question_type is not None and anns['question_type'] not in self.config.question_type:
                continue
            if self.config.answer_encode_format == 'softmax':
                one_word_answer = [ax['answer'] for ax in anns['answers'] if
                                   len(text_to_word_sequence(ax['answer'])) == 1 and ax['answer'] in ('yes', 'no')]
                if not one_word_answer:
                    continue
                counter = Counter(one_word_answer)
                [(w, v)] = counter.most_common(1)
                answers_ls.append([anns['image_id'], anns['question_id'], w])
                self.answer_type_set.add(anns['answer_type'])
                self.question_type_set.add(anns['question_type'])
            else:
                [(w, v)] = Counter([ax['answer'] for ax in anns['answers']]).most_common(1)
                answers_ls.append(
                    [anns['image_id'], anns['question_id'], w])
            self._question_ids.add(anns['question_id'])

    # TODO add max answer size

    def parse_question(self, ques=None, questions_ls=None):
        for que in ques['questions']:
            if que['question_id'] not in self._question_ids:
                continue
            ques_ls = text_to_word_sequence(que['question'])
            if len(ques_ls) > self.max_question_size:
                self.max_question_size = len(ques_ls)
            questions_ls.append([que['image_id'], que['question_id'], que['question']])

    def build_vocab(self, questions, answers):
        if self.config.answer_encode_format != 'softmax':
            answers = ' '.join(answers)
            answers = set([x.strip() for x in re.split('([\d\W])', answers) if x.strip()])

        question_sub_answer = questions - answers
        word_list = sorted(list(answers)) + sorted(list(question_sub_answer))
        print('complete create question and answer word set')
        self.word_size = len(word_list)

        self.answer_word_size = len(answers)
        self.index2words, self.words2index = self.parser_text_to_vocab(word_list)

    def encode_answer(self, answer):
        # assert len(answers) == 10, 'the answers don\'t has 10'
        if self.config.answer_encode_format == 'softmax':
            return self.words2index[answer]
        else:
            return [self.words2index[x.strip()] for x in re.split('([\d\W])', answer) if x.strip()]

    def decode_answer(self, answer_encoder, size):
        """
        :param answer_encoder:
        :param size: the number of answer you want to know
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
        encoder = [self.words2index[w] for w in text_to_word_sequence(question)]
        question_encoder = [0 for i in range(self.max_question_size - len(encoder))]
        question_encoder.extend(encoder)
        return question_encoder

    def parse(self):
        """
        :param answer_only_one_word:
        :param train_questionFile:
        :param train_annFile:
        :type answer_code_format: 'softmax' or 'sqe2sqe'
            when 'softmax' , answer with lager than 1 word will filtered
        """
        print('start loading...')
        train_ann = json.load(open(self.config.train_annFile, 'r'))
        train_ques = json.load(open(self.config.train_questionFile, 'r'))
        val_ann = json.load(open(self.config.val_annFile, 'r'))
        val_ques = json.load(open(self.config.val_questionFile, 'r'))
        print('load complete')

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
        questions = set(text_to_word_sequence(questions))
        answers = set([x[2] for x in answers_train_ls])
        answers = answers.union(set([x[2] for x in answers_val_ls]))

        self.build_vocab(questions, answers)
        print('complete build vocab')

        questionId_answer_train_df = pd.DataFrame(data=answers_train_ls, columns=['image_id', 'question_id', 'answer'])
        questionId_answer_val_df = pd.DataFrame(data=answers_val_ls, columns=['image_id', 'question_id', 'answer'])
        questionId_question_train_df = pd.DataFrame(data=questions_train_ls,
                                                    columns=['image_id', 'question_id', 'question'])
        questionId_question_val_df = pd.DataFrame(data=questions_val_ls,
                                                  columns=['image_id', 'question_id', 'question'])

        questionId_answer_train_df['answer'] = questionId_answer_train_df['answer'].apply(self.encode_answer)
        questionId_answer_val_df['answer'] = questionId_answer_val_df['answer'].apply(self.encode_answer)

        questionId_question_train_df['question'] = questionId_question_train_df['question'].apply(self.encode_question)
        questionId_question_val_df['question'] = questionId_question_val_df['question'].apply(self.encode_question)

        self.train_data = pd.merge(questionId_question_train_df, questionId_answer_train_df,
                                   on=['image_id', 'question_id']).drop(['question_id'], axis=1)
        self.val_data = pd.merge(questionId_question_val_df, questionId_answer_val_df,
                                 on=['image_id', 'question_id']).drop(['question_id'], axis=1)
        self.train_sample_size = len(questionId_answer_train_df)
        self.val_sample_size = len(questionId_answer_val_df)
        print('train_sample_size:%d\n val_sample_size:%d ' % (self.train_sample_size, self.val_sample_size))

    # parse map between image_id and question_id

    def info(self):
        info = ''
        info += "word size:%s\n" % self.word_size
        info += "answer size:%s\n" % self.answer_word_size
        info += "train size: %s\n" % self.train_sample_size
        info += "val size: %s\n" % self.val_sample_size
        info += 'question_type:\n' + str(self.question_type_set) + '\n'
        info += 'answer_type:\n' + str(self.answer_type_set)
        print(info)

    def save_result(self):
        self.train_data.to_csv(self.config.train_data_file)
        self.val_data.to_csv(self.config.val_data_file)

    def save_data(self):
        with open(self.config.save_data_file, 'bw') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    config = Config()
    data = DataParser(config)
    data.parse()
    # data.save_result()
    data.info()
# generate_test = data.generate_data(100, data='val')
# generate_train = data.generate_data(100, data='train')
# [a, b], c = next(generate_train)
