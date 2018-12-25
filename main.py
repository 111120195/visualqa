import os

from keras.callbacks import ModelCheckpoint, EarlyStopping

from config import Config
from data_generater import DataGenerate
from DMNPlus import VqaModel
from keras.callbacks import LearningRateScheduler

config = Config()
data = DataGenerate(config)
lr_base = config.lr
epochs = config.epochs
lr_power = config.lr_power
epoch_decay = config.epoch_decay


def lr_scheduler(epoch):
	lr = float(lr_base) * (float(lr_power) ** (epoch / epoch_decay))
	print('lr: %f' % lr)
	return lr


def train_model():
	train = data.generate_data(baseline=False, data_type='train')
	val = data.generate_data(baseline=False, data_type='val')

	data_info = data.get_data_info()
	steps_per_epoch = data_info['steps_per_epoch']
	validation_steps = data_info['validation_steps']

	model = VqaModel(config, data_info).build_model()
	print(model.summary())
	# TODO select fit parameter
	save_dir = r'./model'
	model_file = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
	checkpoint = ModelCheckpoint(os.path.join(save_dir, model_file), monitor='val_loss', verbose=1,
								 save_best_only=True,
								 mode='min', period=1)
	early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

	scheduler = LearningRateScheduler(lr_scheduler)

	model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=config.epochs, validation_data=val,
						validation_steps=validation_steps, callbacks=[checkpoint, early_stopping, scheduler])


def predict():
	pass


if __name__ == '__main__':
	train_model()
