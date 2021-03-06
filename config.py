class Config(object):
	def __init__(self):
		# data file
		self.train_annFile = r'../data/v2_mscoco_train2014_annotations.json'
		self.train_questionFile = r'../data/v2_OpenEnded_mscoco_train2014_questions.json'
		self.val_annFile = r'../data/v2_mscoco_val2014_annotations.json'
		self.val_questionFile = r'../data/v2_OpenEnded_mscoco_val2014_questions.json'
		self.train_img_dir = r'../data/train2014/COCO_train2014_'
		self.val_img_dir = r'../data/val2014/COCO_val2014_'

		# filter condition
		self.answer_type = ['other']
		# {'other', 'number', 'yes/no'}
		self.question_type = None
		# {'what is on the', 'are the', 'is it', 'is the person', 'can you', 'what is the name', 'how many people are',
		#  'what are', 'how many people are in', 'is the man', 'do you', 'is this', 'are they', 'is he', 'why is the',
		#  'where are the', 'does the', 'is there', 'could', 'which', 'is that a', 'where is the', 'what are the', 'why',
		#  'what room is', 'are', 'none of the above', 'what is', 'what is the woman', 'what color is', 'is the woman',
		#  'has', 'what is the color of the', 'is this person', 'what number is', 'how many', 'is this a', 'what type of',
		#  'what animal is', 'does this', 'are these', 'how', 'what is in the', 'are there', 'what is the man', 'is the',
		#  'what does the', 'who is', 'what is the', 'what color', 'do', 'what brand', 'what', 'what color is the',
		#  'are there any', 'was', 'is this an', 'what is the person', 'is there a', 'what color are the', 'what kind of',
		#  'what time', 'is', 'what sport is', 'what is this'}

		# data save directory
		self.train_data_file = r"../data/train_data.csv"
		self.val_data_file = r"../data/val_data.csv"
		self.vocab2index_dict = r'../data/w2i.txt'
		self.index2vocab_dict = r'../data/i2w.txt'
		self.save_data_file = r'../data/data.pkl'
		self.train_image_feature_dir = r'../data/train_image_feature'
		self.val_image_feature_dir = r'../data/val_image_feature'

		self.base_train_image_feature_dir = r'../data/base_train_image_feature'
		self.base_val_image_feature_dir = r'../data/base_val_image_feature'

		# answer encode format
		self.answer_encode_format = 'softmax'

		# model architecture hyper parameter
		self.image_input_shape = (14 * 14, 512)
		self.word_embedding_size = 256
		self.gru_hidden_size = 512
		self.drop_out_rate = 0.5
		self.episodic_memory_hidden_size = 512
		self.memory_size = 512
		self.answer_output_hidden_size = 512

		# optimizer parameter
		self.optimizer = 'adam' # 'sgd', 'rmsprop'
		self.lr = 0.001
		self.lr_power = 0.5
		self.epoch_decay = 5

		# fit parameter
		self.batch_size = 128
		self.epochs = 10
