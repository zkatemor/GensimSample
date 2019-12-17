from keras.layers.core import Dense, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Activation
from keras.utils import to_categorical
from keras.models import save_model, load_model
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import RegexpTokenizer
import collections
import gensim.downloader as api
import pymorphy2
import pickle
from sklearn.utils import shuffle


EMBED_SIZE = 300
NUM_FILTERS = 512
NUM_WORDS = 3
BATCH_SIZE = 32
NUM_EPOCHS = 15


def fit(requests, category_ids):  # messages - LIST!!! предложений, category_ids - LIST!!! id категорий
    requests, category_ids = shuffle(requests, category_ids)

    sentences = np.array([tokenize(request) for request in requests])
    counter, maxlen = get_counter_and_maxlen(sentences)
    vocab_sz = len(counter) + 1  # кол-во различных слов в sentences
    embedding_weights = get_embedding_weights(counter, vocab_sz)

    X_train = get_X_train(sentences, maxlen)
    y_train = to_categorical(category_ids)
    print(y_train)

    keras_model = get_model(vocab_sz, maxlen, embedding_weights, classes_count=len(np.unique(category_ids)))
    history = keras_model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    callbacks=[ModelCheckpoint('keras_model.h5', save_best_only=True)],
                    validation_split=0.2, verbose=1)
    return history


def predict(request):  # request - строка, запрос
    keras_model = load_model('keras_model.h5')

    tokenizer = get_tokenizer()

    sentences = [tokenize(request)]
    X_predict = tokenizer.texts_to_sequences(sentences)
    X_predict = pad_sequences(X_predict, maxlen=get_maxlen())

    prediction = keras_model.predict_classes(X_predict)

    return prediction[0]


def get_embedding_weights(counter, vocab_sz):
    word2vec_model = api.load("word2vec-ruscorpora-300")

    embedding_weights = np.zeros(
        (vocab_sz, EMBED_SIZE))  # создаём матрицу размером размерность словаря*размерность вектора слова
    index = 0
    sorted_counter = counter.most_common()  # сортируем слова по частоте встречаемости
    for word in sorted_counter:  # для каждого слова из нашего словаря задаём вектор из model в матрицу
        try:
            embedding_weights[index, :] = word2vec_model[word[0]]
            index += 1
        except KeyError:  # если нет слова в словаре model
            index += 1
            pass

    return embedding_weights


def get_counter_and_maxlen(sentences):
    counter = collections.Counter()

    # считаем максимальную длину предложений, а также частоту всех слов предложений, считанных из файла
    maxlen = 0
    for words in sentences:
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            counter[word] += 1

    save_maxlen(maxlen)

    return counter, maxlen


def save_maxlen(maxlen):
    f = open('maxlen.bin', 'w')  # открытие в режиме записи
    f.write(str(maxlen))
    f.close()


def get_maxlen():
    f = open('maxlen.bin', 'r')
    maxlen = int(f.read())  # чтение
    f.close()

    return maxlen


def get_X_train(sentences, maxlen):
    # Создание единого словаря (слово -> число) для преобразования на основе списка текстов sentences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    X_train = tokenizer.texts_to_sequences(sentences)  # заменяем слова каждого предложения на числа
    X_train = pad_sequences(X_train, maxlen=maxlen)  # уравниваем все предложения до размера maxlen
    save_tokenizer(tokenizer)

    return X_train


def save_tokenizer(tokenizer):
    # сохранение токенайзера
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_tokenizer():
    # открытие токенайзера из файла
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def tokenize(request):
    morph = pymorphy2.MorphAnalyzer()  # для перевода в нормальную форму
    regex_tokenizer = RegexpTokenizer('[а-яА-ЯЁё]+')
    words = regex_tokenizer.tokenize(request.lower())
    stop_words = set(stopwords.words("russian"))
    without_stop_words = [(morph.parse(w)[0]).normal_form for w in words if w not in stop_words and len(w) > 1]
    output = [add_part_of_speech(morph, word) for word in without_stop_words]

    return output


def add_part_of_speech(morph, word):
    p = morph.parse(word)[0]
    word += '_' + str(p.tag.POS)
    return word


# Создание модели
def get_model(vocab_sz, maxlen, embedding_weights, classes_count):
    model = Sequential()
    model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen,
                        weights=[embedding_weights],
                        trainable=True))

    model.add(Dropout(0.2))

    model.add(Conv1D(50,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(classes_count, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model