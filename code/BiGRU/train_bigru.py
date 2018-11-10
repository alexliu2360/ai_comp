# -*- coding: utf-8 -*-
import os
import time
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from mxnet.contrib import text

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import random

random.seed = 42
import pandas as pd
from tensorflow import set_random_seed

set_random_seed(42)
from keras.preprocessing import text as keras_text
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score
from keras.layers import *
from .classifier_bigru import TextClassifier
from gensim.models.keyedvectors import KeyedVectors
import pickle
import gc

from utils import getClassification

from config import preprocess_data_path, preprocess_train_data_fn, preprocess_validate_data_fn, bigru_models_path, \
    word2vec_chars_path, word2vec_chars_fn, column_batch_map, column_list, bigru_embeddings_matrix_path, \
    bigru_embeddings_matrix_suffix, bigru_embeddings_matrix_fn, tokenizer_bigru_path, tokenizer_bigru_fn, \
    own_word2vec_fn, own_withsw_char_word2vec_fn


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


class BigruOwnW2v:
    def __init__(self, maxlen=1200):
        self.maxlen = maxlen
        self.preprocess_train_data_file = preprocess_data_path + preprocess_train_data_fn
        self.preprocess_validate_data_file = preprocess_data_path + preprocess_validate_data_fn
        # self.word2vec_chars_file = word2vec_chars_path + own_word2vec_fn
        self.word2vec_chars_file = word2vec_chars_path + own_withsw_char_word2vec_fn
        self.tokenizer_bigru_file = tokenizer_bigru_path + tokenizer_bigru_fn
        self._prepare_data()

    def _prepare_data(self):
        self.prep_train_data = pd.read_csv(self.preprocess_train_data_file)
        self.prep_validate_data = pd.read_csv(self.preprocess_validate_data_file)
        self.prep_train_data["content"] = self.prep_train_data.apply(lambda x: eval(x[1]), axis=1)
        self.prep_validate_data["content"] = self.prep_validate_data.apply(lambda x: eval(x[1]), axis=1)

        # 加载分词器
        if not os.path.exists(self.tokenizer_bigru_file):
            self.tokenizer = keras_text.Tokenizer(num_words=None)
            self.tokenizer.fit_on_texts(self.prep_train_data["content"].values)
            with open(self.tokenizer_bigru_file, 'wb') as f:
                pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.tokenizer_bigru_file, 'rb') as f:
                self.tokenizer = pickle.load(f)

        self.word_index = self.tokenizer.word_index

        # 开始load预训练的词向量
        print('load embedding_matrix...')
        start_load_embeddings_matrix = time.time()
        w2_model = KeyedVectors.load_word2vec_format(self.word2vec_chars_file, binary=True,
                                                     encoding='utf8',
                                                     unicode_errors='ignore')
        self.embeddings_matrix = np.zeros((len(self.word_index) + 1, w2_model.vector_size))

        for word, i in self.word_index.items():
            if word in w2_model:
                self.embeddings_matrix[i] = w2_model[word]
        print('load embeddings_matrix time used %f s...' % (time.time() - start_load_embeddings_matrix))

        # 准备训练数据
        X_train = self.prep_train_data["content"].values
        X_validation = self.prep_validate_data["content"].values

        list_tokenized_train = self.tokenizer.texts_to_sequences(X_train)
        self.input_train = sequence.pad_sequences(list_tokenized_train, maxlen=self.maxlen)

        list_tokenized_validation = self.tokenizer.texts_to_sequences(X_validation)
        self.input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=self.maxlen)

    def train(self, model_index, batch_size=128, epochs=8):
        # 模型索引 其实就是20个大类别
        model_index = int(model_index)
        y_train = pd.get_dummies(self.prep_train_data[column_list[model_index]])[[-2, -1, 0, 1]].values
        y_val = pd.get_dummies(self.prep_validate_data[column_list[model_index]])[[-2, -1, 0, 1]].values

        print("model" + str(model_index) + ' start...')
        model = TextClassifier().model(self.embeddings_matrix, self.maxlen, self.word_index, 4)
        file_path = bigru_models_path + "model_" + str(column_batch_map[model_index]) + "_{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(file_path, verbose=1, save_weights_only=True)
        metrics = Metrics()
        callbacks_list = [checkpoint, metrics]
        model.fit(self.input_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(self.input_validation, y_val), callbacks=callbacks_list, verbose=1)
        print('model' + str(model_index) + ' ending...')


class BigruFasttextW2v:
    def __init__(self, maxlen=1200):
        self.maxlen = maxlen
        self.preprocess_train_data_file = preprocess_data_path + preprocess_train_data_fn
        self.preprocess_validate_data_file = preprocess_data_path + preprocess_validate_data_fn
        self.word2vec_chars_file = word2vec_chars_path + word2vec_chars_fn
        self._prepare_data()

    def _prepare_data(self):

        self.data = pd.read_csv(self.preprocess_train_data_file)
        self.validation = pd.read_csv(self.preprocess_validate_data_file)
        self.data["content"] = self.data.apply(lambda x: eval(x[1]), axis=1)
        self.validation["content"] = self.validation.apply(lambda x: eval(x[1]), axis=1)
        tokenizer_bigru_file = tokenizer_bigru_path + tokenizer_bigru_fn
        if not os.path.exists(tokenizer_bigru_file):
            self.tokenizer = keras_text.Tokenizer(num_words=None)
            self.tokenizer.fit_on_texts(self.data["content"].values)
            with open(tokenizer_bigru_file, 'wb') as f:
                pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(tokenizer_bigru_file, 'rb') as f:
                self.tokenizer = pickle.load(f)

        self.word_index = self.tokenizer.word_index
        embedding_matrix_file = bigru_embeddings_matrix_path + bigru_embeddings_matrix_fn
        print(embedding_matrix_file + '.' + bigru_embeddings_matrix_suffix)
        if not os.path.exists(embedding_matrix_file + '.' + bigru_embeddings_matrix_suffix):
            print('make embedding_matrix...')
            start_create_wiki_zh_vec_time = time.time()
            wiki_zh_vec = text.embedding.create('fasttext', pretrained_file_name='wiki.zh.vec')
            end_create_wiki_zh_vec_time = time.time()
            print(
                'create wiki_zh_vec time used %f s...' % (end_create_wiki_zh_vec_time - start_create_wiki_zh_vec_time))
            self.embeddings_matrix = np.zeros((len(self.word_index) + 1, wiki_zh_vec.vec_len))
            for word, i in self.word_index.items():
                if word in wiki_zh_vec.idx_to_token:
                    self.embeddings_matrix[i] = wiki_zh_vec.get_vecs_by_tokens(word).asnumpy()
            end_create_embeddings_matrix_time = time.time()
            print(
                'create embeddings_matrix time used %f s...' % (
                end_create_embeddings_matrix_time - end_create_wiki_zh_vec_time))
            np.save(embedding_matrix_file, self.embeddings_matrix)
            print(
                'save embeddings_matrix time used %f s...' % (
                    time.time() - end_create_embeddings_matrix_time))
        else:
            print('load embedding_matrix...')
            start_load_embeddings_matrix = time.time()
            self.embeddings_matrix = np.load(embedding_matrix_file + '.' + bigru_embeddings_matrix_suffix)
            print(
                'load embeddings_matrix time used %f s...' % (time.time() - start_load_embeddings_matrix))
        X_train = self.data["content"].values
        X_validation = self.validation["content"].values

        list_tokenized_train = self.tokenizer.texts_to_sequences(X_train)
        self.input_train = sequence.pad_sequences(list_tokenized_train, maxlen=self.maxlen)

        list_tokenized_validation = self.tokenizer.texts_to_sequences(X_validation)
        self.input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=self.maxlen)

    def train(self, model_index, batch_size=128, epochs=5):
        # 模型索引 其实就是20个大类别
        model_index = int(model_index)
        y_train = pd.get_dummies(self.data[column_list[model_index]])[[-2, -1, 0, 1]].values
        y_val = pd.get_dummies(self.validation[column_list[model_index]])[[-2, -1, 0, 1]].values

        print("model" + str(model_index))
        model = TextClassifier().model(self.embeddings_matrix, self.maxlen, self.word_index, 4)
        file_path = bigru_models_path + "model_" + str(column_batch_map[model_index]) + "_{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(file_path, verbose=1, save_weights_only=True)
        metrics = Metrics()
        callbacks_list = [checkpoint, metrics]
        history = model.fit(self.input_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(self.input_validation, y_val), callbacks=callbacks_list, verbose=1)
        # del model
        # del history
        print(history)
        # gc.collect()
        # K.clear_session()


class BiGRU:
    def __init__(self, maxlen=1200):
        self._prepare_data()
        self.maxlen = maxlen
        self.preprocess_train_data_file = preprocess_data_path + preprocess_train_data_fn
        self.preprocess_validate_data_file = preprocess_data_path + preprocess_validate_data_fn
        self.word2vec_chars_file = word2vec_chars_path + word2vec_chars_fn

    def _prepare_data(self):
        self.data = pd.read_csv(self.preprocess_train_data_file)
        self.validation = pd.read_csv(self.preprocess_validate_data_file)
        self.data["content"] = self.data.apply(lambda x: eval(x[1]), axis=1)
        self.validation["content"] = self.validation.apply(lambda x: eval(x[1]), axis=1)

        tokenizer = keras_text.Tokenizer(num_words=None)
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

    def train(self, model_index, batch_size=128, epochs=10):
        # 模型索引 其实就是20个大类别
        model_index = int(model_index)
        y_train = pd.get_dummies(self.data[column_list[model_index]])[[-2, -1, 0, 1]].values
        y_val = pd.get_dummies(self.validation[column_list[model_index]])[[-2, -1, 0, 1]].values

        print("model" + str(model_index))
        model = TextClassifier().model(self.embeddings_matrix, self.maxlen, self.word_index, 4)
        file_path = bigru_models_path + "model_" + str(column_batch_map[model_index]) + "_{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(file_path, verbose=1, save_weights_only=True)
        metrics = Metrics()
        callbacks_list = [checkpoint, metrics]
        history = model.fit(self.input_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(self.input_validation, y_val), callbacks=callbacks_list, verbose=2)
        # del model
        # del history
        print(history)
        # gc.collect()
        K.clear_session()

# for word, i in word_index.items():
#     if word in wiki_zh_vec.idx_to_token:
#         print('[%d]'%(i))
#         embeddings_matrix[i] = wiki_zh_vec.get_vecs_by_tokens(word).asnumpy()
