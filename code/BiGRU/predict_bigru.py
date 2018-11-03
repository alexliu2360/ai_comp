from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from config import original_data_path, \
    word2vec_chars_path, word2vec_chars_fn, preprocess_data_path, \
    bigru_models_path, tokenizer_bigru_fn, tokenizer_bigru_path, column_batch_map, \
    column_list, preprocess_testa_data_fn, original_test_fn, predict_data_path, predict_bigru_fn, predict_bigru_prob_fn

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import gc
import pandas as pd
import pickle
import numpy as np

np.random.seed(16)
from tensorflow import set_random_seed

set_random_seed(16)
from keras.layers import *
from keras.preprocessing import sequence
from gensim.models.keyedvectors import KeyedVectors
from .classifier_bigru import TextClassifier


def getClassification(arr):
    arr = list(arr)
    if arr.index(max(arr)) == 0:
        return -2
    elif arr.index(max(arr)) == 1:
        return -1
    elif arr.index(max(arr)) == 2:
        return 0
    else:
        return 1


def part_bigru_predict(index_list, maxlen=1200, epochs=10, predict_fn=predict_bigru_fn,
                       predict_prob_fn=predict_bigru_prob_fn, mode='w'):
    with open(tokenizer_bigru_path + tokenizer_bigru_fn, 'rb') as handle:
        model_dir = bigru_models_path
        tokenizer = pickle.load(handle)
        word_index = tokenizer.word_index
        validation = pd.read_csv(preprocess_data_path + preprocess_testa_data_fn)
        validation["content"] = validation.apply(lambda x: eval(x[1]), axis=1)
        X_test = validation["content"].values
        list_tokenized_validation = tokenizer.texts_to_sequences(X_test)
        input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=maxlen)
        w2_model = KeyedVectors.load_word2vec_format(word2vec_chars_path + word2vec_chars_fn, binary=True,
                                                     encoding='utf8',
                                                     unicode_errors='ignore')

        embeddings_matrix = np.zeros((len(word_index) + 1, w2_model.vector_size))
        for word, i in word_index.items():
            if word in w2_model:
                embedding_vector = w2_model[word]
            else:
                embedding_vector = None
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector

        submit = pd.read_csv(original_data_path + original_test_fn)
        submit_prob = pd.read_csv(original_data_path + original_test_fn)

        for index in index_list:
            print("[bigru model %d predict...]" % index)
            model = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
            model.load_weights(model_dir + "model_" + column_batch_map[index] + "_" + str(epochs) + ".hdf5")
            submit[column_list[index]] = list(map(getClassification, model.predict(input_validation)))
            submit_prob[column_list[index]] = list(model.predict(input_validation))
            del model
            gc.collect()
            K.clear_session()

        submit.to_csv(predict_data_path + predict_fn, index=None, mode=mode)
        submit_prob.to_csv(predict_data_path + predict_prob_fn, index=None, mode=mode)


def all_bigru_predict(maxlen=1200, epochs=10):
    with open(tokenizer_bigru_path + tokenizer_bigru_fn, 'rb') as handle:
        model_dir = bigru_models_path
        tokenizer = pickle.load(handle)
        word_index = tokenizer.word_index
        validation = pd.read_csv(preprocess_data_path + preprocess_testa_data_fn)
        validation["content"] = validation.apply(lambda x: eval(x[1]), axis=1)
        X_test = validation["content"].values
        list_tokenized_validation = tokenizer.texts_to_sequences(X_test)
        input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=maxlen)
        w2_model = KeyedVectors.load_word2vec_format(word2vec_chars_path + word2vec_chars_fn, binary=True,
                                                     encoding='utf8',
                                                     unicode_errors='ignore')

        embeddings_matrix = np.zeros((len(word_index) + 1, w2_model.vector_size))

        for word, i in word_index.items():
            if word in w2_model:
                embedding_vector = w2_model[word]
            else:
                embedding_vector = None
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector

        submit = pd.read_csv(original_data_path + original_test_fn)
        submit_prob = pd.read_csv(original_data_path + original_test_fn)

        for index in range(1, 21):
            model = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
            model.load_weights(model_dir + "model_" + column_batch_map[index] + "_" + str(epochs) + ".hdf5")
            submit[column_list[index]] = list(map(getClassification, model.predict(input_validation)))
            submit_prob[column_list[index]] = list(model.predict(input_validation))
            del model
            gc.collect()
            K.clear_session()

        submit.to_csv(predict_data_path + predict_bigru_fn, index=None)
        submit_prob.to_csv(predict_data_path + predict_bigru_prob_fn, index=None)
