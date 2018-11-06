# -*- coding: utf-8 -*-
import os
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
    bigru_embeddings_matrix_suffix, bigru_embeddings_matrix_fn


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

        tokenizer = keras_text.Tokenizer(num_words=None)
        tokenizer.fit_on_texts(self.data["content"].values)
        with open('tokenizer_char.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.word_index = tokenizer.word_index
        embedding_matrix_file = bigru_embeddings_matrix_path + bigru_embeddings_matrix_fn
        print('make or load embedding_matrix...')
        if not os.path.exists(embedding_matrix_file):
            print('make embedding_matrix...')
            wiki_zh_vec = text.embedding.create('fasttext', pretrained_file_name='wiki.zh.vec')
            self.embeddings_matrix = np.zeros((len(self.word_index) + 1, wiki_zh_vec.vec_len))
            for word, i in self.word_index.items():
                if word in wiki_zh_vec.idx_to_token:
                    self.embeddings_matrix[i] = wiki_zh_vec.get_vecs_by_tokens(word).asnumpy()
            np.save(embedding_matrix_file, self.embeddings_matrix)
        else:
            print('load embedding_matrix...')
            self.embeddings_matrix = np.load(embedding_matrix_file+'.npy')

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

# for word, i in word_index.items():
#     if word in wiki_zh_vec.idx_to_token:
#         print('[%d]'%(i))
#         embeddings_matrix[i] = wiki_zh_vec.get_vecs_by_tokens(word).asnumpy()
