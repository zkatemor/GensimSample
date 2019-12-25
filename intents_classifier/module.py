import gensim
import keras
import nltk
import sklearn
import numpy
import collections
import pymorphy2
import pickle
import gensim.downloader
import os.path
import typing

__EMBED_SIZE = 300
__NUM_FILTERS = 512
__NUM_WORDS = 3
__BATCH_SIZE = 32
__NUM_EPOCHS = 15
__keras_model_path = 'keras_model.h5'
__tokenizer_path = 'tokenizer.pickle'
__maxlen_path = 'maxlen.bin'


# Обучение модели. ВХОД: requests - список (list) всех запросов, category_ids - список (list) всех id запросов
# ВЫХОД: история обучения модели
def fit(requests: typing.List[str], category_ids: typing.List[str]) -> keras.callbacks.callbacks.History():
    __save_category_ids(category_ids)  # сохраняем исходные id

    category_ids = __get_consecutive_indices(category_ids)  # получаем последовательные id для корректной работы
    requests, category_ids = sklearn.utils.shuffle(requests, category_ids)  # перемешиваем данные

    sentences = numpy.array([__tokenize(request) for request in requests])  # предобработка
    counter, maxlen = __get_counter_and_maxlen(sentences)
    vocab_sz = len(counter) + 1  # кол-во различных слов в sentences
    embedding_weights = __get_embedding_weights(counter, vocab_sz)
    X_train = __get_X_train(sentences, maxlen)
    y_train = keras.utils.to_categorical(category_ids)
    keras_model = __get_model(vocab_sz, maxlen, embedding_weights, classes_count=len(numpy.unique(category_ids)))

    history = keras_model.fit(X_train, y_train, batch_size=__BATCH_SIZE,
                              epochs=__NUM_EPOCHS,
                              callbacks=[keras.callbacks.ModelCheckpoint('keras_model.h5', save_best_only=True)],
                              validation_split=0.2, verbose=0)

    return history


# Предсказание id запроса. ВХОД: request - строка, запрос
# ВЫХОД: id запроса request (>= 0 или -1, если модель не была обучена)
def predict(request: str) -> int:
    if os.path.isfile(__keras_model_path) and os.path.isfile(__tokenizer_path) and os.path.isfile(__maxlen_path):
        keras_model = keras.models.load_model(__keras_model_path)  # получаем сохранённую ранее модель

        tokenizer = __get_tokenizer()  # получаем сохранённый ранее токенайзер

        sentences = [__tokenize(request)]
        X_predict = tokenizer.texts_to_sequences(sentences)
        X_predict = keras.preprocessing.sequence.pad_sequences(X_predict, maxlen=__get_maxlen())

        prediction = keras_model.predict_classes(X_predict)
        indices_unique = numpy.unique(__get_category_ids())  # получаем исходные id

        return indices_unique[prediction[0]]
    else:
        print('Do not exists keras_model or another necessary files. For beginning, train the model (use fit)')

        return -1


# Получение последовательных id из категорий интентов
def __get_consecutive_indices(category_ids):
    consecutive_indices = []
    unique_id = numpy.unique(category_ids).tolist()

    for id in category_ids:
        consecutive_indices.append(unique_id.index(id))

    return consecutive_indices


# Сохранение исходных id категории
def __save_category_ids(category_ids):
    with open('indices.pickle', 'wb') as f:
        pickle.dump(category_ids, f)


# Получение исходных id категории
def __get_category_ids():
    with open('indices.pickle', 'rb') as f:
        indices = pickle.load(f)

    return indices


# Получение векторов слов из word2vec_model
def __get_embedding_weights(counter, vocab_sz):
    word2vec_model = gensim.downloader.load("word2vec-ruscorpora-300")

    embedding_weights = numpy.zeros(
        (vocab_sz, __EMBED_SIZE))  # создаём матрицу размером размерность словаря*размерность вектора слова
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


# Получение всех слов из предложений sentences + их частоты, максимальной длины предложений
def __get_counter_and_maxlen(sentences):
    counter = collections.Counter()

    maxlen = 0
    for words in sentences:
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            counter[word] += 1

    __save_maxlen(maxlen)

    return counter, maxlen


# Методы сохранения, получения maxlen

def __save_maxlen(maxlen):
    f = open(__maxlen_path, 'w')
    f.write(str(maxlen))
    f.close()


def __get_maxlen():
    f = open(__maxlen_path, 'r')
    maxlen = int(f.read())
    f.close()

    return maxlen


# Преобразование запросов sentences
def __get_X_train(sentences, maxlen):
    # Создание единого словаря (слово -> число) для преобразования на основе списка текстов sentences
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    X_train = tokenizer.texts_to_sequences(sentences)  # заменяем слова каждого предложения на числа
    X_train = keras.preprocessing.sequence.pad_sequences(X_train,
                                                         maxlen=maxlen)  # уравниваем все предложения до размера maxlen
    __save_tokenizer(tokenizer)

    return X_train


# Методы сохранения, получения токенайзера

def __save_tokenizer(tokenizer):
    # сохранение токенайзера
    with open(__tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def __get_tokenizer():
    # открытие токенайзера из файла
    with open(__tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


# Предобработка запроса
def __tokenize(request):
    morph = pymorphy2.MorphAnalyzer()  # для перевода в нормальную форму
    regex_tokenizer = nltk.tokenize.RegexpTokenizer('[а-яА-ЯЁё]+')
    words = regex_tokenizer.tokenize(request.lower())
    stop_words = set(nltk.corpus.stopwords.words("russian"))
    without_stop_words = [(morph.parse(w)[0]).normal_form for w in words if w not in stop_words and len(w) > 1]
    output = [__add_part_of_speech(morph, word) for word in without_stop_words]

    return output


# Добавление к слову части речи
def __add_part_of_speech(morph, word):
    p = morph.parse(word)[0]
    word += '_' + str(p.tag.POS)
    return word


# Создание модели
def __get_model(vocab_sz, maxlen, embedding_weights, classes_count):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(vocab_sz, __EMBED_SIZE, input_length=maxlen,
                                     weights=[embedding_weights],
                                     trainable=True))

    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(50,
                                  3,
                                  padding='valid',
                                  activation='relu',
                                  strides=1))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(250))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(classes_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
