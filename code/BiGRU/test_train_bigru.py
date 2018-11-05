# -*- coding: utf-8 -*-
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras
from keras import Model
from keras.layers import *
from utils.JoinAttLayer import Attention

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import random

random.seed = 42
import pandas as pd
from tensorflow import set_random_seed

set_random_seed(42)
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score
from keras.layers import *
from gensim.models.keyedvectors import KeyedVectors
import pickle
import gc

from utils import getClassification

from config import preprocess_data_path, preprocess_train_data_fn, preprocess_validate_data_fn, bigru_models_path, \
    word2vec_chars_path, word2vec_chars_fn, column_batch_map, column_list
from config import test_train_data_file, test_validate_data_file


class TextClassifier():
    def model(self, embeddings_matrix, maxlen, word_index, num_class):
        inp = Input(shape=(maxlen,))
        encode = Bidirectional(CuDNNGRU(128, return_sequences=True))
        encode2 = Bidirectional(CuDNNGRU(128, return_sequences=True))
        attention = Attention(maxlen)
        x_4 = Embedding(len(word_index) + 1,
                        embeddings_matrix.shape[1],
                        weights=[embeddings_matrix],
                        input_length=maxlen,
                        trainable=True)(inp)
        x_3 = SpatialDropout1D(0.2)(x_4)
        x_3 = encode(x_3)
        x_3 = Dropout(0.2)(x_3)
        x_3 = encode2(x_3)
        x_3 = Dropout(0.2)(x_3)
        avg_pool_3 = GlobalAveragePooling1D()(x_3)
        max_pool_3 = GlobalMaxPooling1D()(x_3)
        attention_3 = attention(x_3)
        x = keras.layers.concatenate([avg_pool_3, max_pool_3, attention_3], name="fc")
        x = Dense(num_class, activation="sigmoid")(x)

        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        model = Model(inputs=inp, outputs=x)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=adam)
        return model


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = list(map(getClassification, self.model.predict(self.validation_data[0])))
        val_targ = list(map(getClassification, self.validation_data[1]))
        _val_f1 = f1_score(val_targ, val_predict, average="macro")
        _val_recall = recall_score(val_targ, val_predict, average="macro")
        _val_precision = precision_score(val_targ, val_predict, average="macro")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(_val_f1, _val_precision, _val_recall)
        print("max f1")
        print(max(self.val_f1s))
        return


class Test_BiGRU:
    def __init__(self, train_data_file, validate_data_file, maxlen=1200):
        self.maxlen = maxlen
        self.word2vec_chars_file = word2vec_chars_path + word2vec_chars_fn
        self.train_data_file = train_data_file
        self.validate_data_file = validate_data_file

    def prepare_data(self):
        self.data = pd.read_csv(self.train_data_file)
        self.validation = pd.read_csv(self.validate_data_file)
        self.data["content"] = self.data.apply(lambda x: eval(x[1]), axis=1)
        self.validation["content"] = self.validation.apply(lambda x: eval(x[1]), axis=1)

        tokenizer = text.Tokenizer(num_words=None)
        tokenizer.fit_on_texts(self.data["content"].values)
        with open('tokenizer_char.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.word_index = tokenizer.word_index
        w2_model = KeyedVectors.load_word2vec_format(self.word2vec_chars_file, binary=True,
                                                     encoding='utf8',
                                                     unicode_errors='ignore')
        self.embeddings_matrix = np.zeros((len(self.word_index) + 1, w2_model.vector_size))

        for word, i in self.word_index.items():
            if word in w2_model:
                embedding_vector = w2_model[word]
            else:
                embedding_vector = None
            if embedding_vector is not None:
                self.embeddings_matrix[i] = embedding_vector

        X_train = self.data["content"].values
        X_validation = self.validation["content"].values

        list_tokenized_train = tokenizer.texts_to_sequences(X_train)
        self.input_train = sequence.pad_sequences(list_tokenized_train, maxlen=self.maxlen)

        list_tokenized_validation = tokenizer.texts_to_sequences(X_validation)
        self.input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=self.maxlen)


class BiGRU:
    def __init__(self, maxlen=1200):
        self._prepare_data()
        self.maxlen = maxlen
        self.preprocess_train_data_file = preprocess_data_path + preprocess_train_data_fn
        self.preprocess_validate_data_file = preprocess_data_path + preprocess_validate_data_fn
        self.word2vec_chars_file = word2vec_chars_path + word2vec_chars_fn

    def _prepare_data(self):
        self.train_data = pd.read_csv(self.preprocess_train_data_file)
        self.validation_data = pd.read_csv(self.preprocess_validate_data_file)
        self.train_data["content"] = self.train_data.apply(lambda x: eval(x[1]), axis=1)
        self.validation_data["content"] = self.validation_data.apply(lambda x: eval(x[1]), axis=1)

        tokenizer = text.Tokenizer(num_words=None)
        tokenizer.fit_on_texts(self.train_data["content"].values)
        with open('tokenizer_char.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.word_index = tokenizer.word_index
        w2_model = KeyedVectors.load_word2vec_format(self.word2vec_chars_file, binary=True,
                                                     encoding='utf8',
                                                     unicode_errors='ignore')
        self.embeddings_matrix = np.zeros((len(self.word_index) + 1, w2_model.vector_size))

        # 将每个词对应成相应的词向量 横坐标是词索引，纵坐标是词向量  两个构成词向量矩阵
        '''
        embeddings_matrix:
        0: [0.38, 0.45, ....]
        1: [0.23, 0.69, ....]
        2: [0.72, 0.25, ....]
        3: [0.05, 0.91, ....]
        ...
        '''
        for word, i in self.word_index.items():
            if word in w2_model:
                embedding_vector = w2_model[word]
            else:
                embedding_vector = None
            if embedding_vector is not None:
                self.embeddings_matrix[i] = embedding_vector

        X_train = self.train_data["content"].values
        X_validation = self.validation_data["content"].values

        list_tokenized_train = tokenizer.texts_to_sequences(X_train)
        self.input_train = sequence.pad_sequences(list_tokenized_train, maxlen=self.maxlen)

        list_tokenized_validation = tokenizer.texts_to_sequences(X_validation)
        self.input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=self.maxlen)

    def train(self, model_index, batch_size=128, epochs=10):
        # 模型索引 其实就是20个大类别
        model_index = int(model_index)
        y_train = pd.get_dummies(self.train_data[column_list[model_index]])[[-2, -1, 0, 1]].values
        y_val = pd.get_dummies(self.validation_data[column_list[model_index]])[[-2, -1, 0, 1]].values

        print("model" + str(model_index))
        model1 = TextClassifier().model(self.embeddings_matrix, self.maxlen, self.word_index, 4)
        file_path = bigru_models_path + "model_" + str(column_batch_map[model_index]) + "_{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(file_path, verbose=1, save_weights_only=True)
        metrics = Metrics()
        callbacks_list = [checkpoint, metrics]
        history = model1.fit(self.input_train, y_train, batch_size=batch_size, epochs=epochs,
                             validation_data=(self.input_validation, y_val), callbacks=callbacks_list, verbose=2)
        del model1
        del history
        gc.collect()
        K.clear_session()


if __name__ == '__main__':
    bigru = Test_BiGRU(test_train_data_file, test_validate_data_file)
    bigru.prepare_data()

