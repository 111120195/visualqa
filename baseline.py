import keras.backend as K
from keras import Input, optimizers
from keras.applications import VGG19
from keras.layers import GRU, Dropout, Dense, Embedding, concatenate, Flatten
from keras.models import Model

from config import Config
from data_generater import DataGenerate


def build_model():
    """
    build and complie model
    :param input_v: visual local feature with shape of (14*14, 512)
    :param input_q: question encoded with vector
    :return: model
    """
    input_q = Input(shape=(query_maxlen,))
    vgg19 = VGG19(weights='imagenet', include_top=False)

    layer_name = []
    for layer in vgg19.layers:
        layer.trainable = False
        layer_name.append(layer.name)
    x = vgg19.output

    q = Embedding(input_dim=vocab_size + 1, output_dim=1000, input_length=query_maxlen)(
        input_q)  # (samples, query_maxlen, embedding_dim)
    q = Dropout(0.5)(q)

    input = concatenate([Flatten(x), Flatten(q)])

    answer = Dense(1, activation='sigmoid')(input)
    model = Model([vgg19.input, input_q], answer)

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
