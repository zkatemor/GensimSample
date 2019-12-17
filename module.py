import gensim
import keras
import nltk
import sklearn
import numpy
import collections
import pymorphy2
import pickle
import gensim.downloader


__EMBED_SIZE = 300
__NUM_FILTERS = 512
__NUM_WORDS = 3
__BATCH_SIZE = 32
__NUM_EPOCHS = 15


def fit(requests, category_ids):  # messages - LIST!!! предложений, category_ids - LIST!!! id категорий
    requests, category_ids = sklearn.utils.shuffle(requests, category_ids)

    sentences = numpy.array([__tokenize(request) for request in requests])
    print(sentences[:5])
    counter, maxlen = __get_counter_and_maxlen(sentences)
    vocab_sz = len(counter) + 1  # кол-во различных слов в sentences
    embedding_weights = __get_embedding_weights(counter, vocab_sz)

    X_train = __get_X_train(sentences, maxlen)
    y_train = keras.utils.to_categorical(category_ids)
    print(y_train)

    keras_model = __get_model(vocab_sz, maxlen, embedding_weights, classes_count=len(numpy.unique(category_ids)))
    history = keras_model.fit(X_train, y_train, batch_size=__BATCH_SIZE,
                              epochs=__NUM_EPOCHS,
                              callbacks=[keras.callbacks.ModelCheckpoint('keras_model.h5', save_best_only=True)],
                              validation_split=0.2, verbose=1)
    return history


def predict(request):  # request - строка, запрос
    keras_model = keras.models.load_model('keras_model.h5')

    tokenizer = __get_tokenizer()

    sentences = [__tokenize(request)]
    X_predict = tokenizer.texts_to_sequences(sentences)
    X_predict = keras.preprocessing.sequence.pad_sequences(X_predict, maxlen=__get_maxlen())

    prediction = keras_model.predict_classes(X_predict)

    return prediction[0]


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


def __get_counter_and_maxlen(sentences):
    counter = collections.Counter()

    # считаем максимальную длину предложений, а также частоту всех слов предложений, считанных из файла
    maxlen = 0
    for words in sentences:
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            counter[word] += 1

    __save_maxlen(maxlen)

    return counter, maxlen


def __save_maxlen(maxlen):
    f = open('maxlen.bin', 'w')  # открытие в режиме записи
    f.write(str(maxlen))
    f.close()


def __get_maxlen():
    f = open('maxlen.bin', 'r')
    maxlen = int(f.read())  # чтение
    f.close()

    return maxlen


def __get_X_train(sentences, maxlen):
    # Создание единого словаря (слово -> число) для преобразования на основе списка текстов sentences
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    X_train = tokenizer.texts_to_sequences(sentences)  # заменяем слова каждого предложения на числа
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)  # уравниваем все предложения до размера maxlen
    __save_tokenizer(tokenizer)

    return X_train


def __save_tokenizer(tokenizer):
    # сохранение токенайзера
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def __get_tokenizer():
    # открытие токенайзера из файла
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer

def __tokenize(request):
    print(request)
    morph = pymorphy2.MorphAnalyzer()  # для перевода в нормальную форму
    regex_tokenizer = nltk.tokenize.RegexpTokenizer('[а-яА-ЯЁё]+')
    words = regex_tokenizer.tokenize(request.lower())
    stop_words = set(nltk.corpus.stopwords.words("russian"))
    without_stop_words = [(morph.parse(w)[0]).normal_form for w in words if w not in stop_words and len(w) > 1]
    output = [__add_part_of_speech(morph, word) for word in without_stop_words]

    return output


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
