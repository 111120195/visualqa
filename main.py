from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

from config import Config
from data_generater import DataGenerate
from model import VqaModel


def train_model():
    config = Config()
    data = DataGenerate(config)
    train = data.generate_data(baseline=False, data_type='train')
    val = data.generate_data(baseline=False, data_type='val')

    data_info = data.get_data_info()
    steps_per_epoch = data_info['steps_per_epoch']
    validation_steps = data_info['validation_steps']
    #

    model = VqaModel(config, data_info).build_model()
    print(model.summary())
    # TODO select fit parameter
    # model.load_weights('model.h5')
    #
    # checkpoint = ModelCheckpoint('vqa.{epoch:02d-{val_loss:.2f}}.h5', monitor='val_loss', verbose=1,
    #                              save_best_only=True,
    #                              mode='min', period=1)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    #
    # model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=100, validation_data=val,
    #                     validation_steps=validation_steps, callbacks=[checkpoint, early_stopping])
    #
    # model.save('vqa.h5')


def predict():
    pass


if __name__ == '__main__':
    train_model()
