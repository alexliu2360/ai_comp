# -*- coding: utf-8 -*-
import random
import re

from gensim.corpora import Dictionary
import keras
from keras_preprocessing import sequence

random.seed = 16
import pandas as pd
from gensim.models.word2vec import Word2Vec
import jieba
# from config import original_data_path, preprocess_data_path, word2vec_chars_path, word2vec_chars_fn \
#     , original_validate_fn, original_test_fn, original_train_fn, preprocess_validate_data_fn, preprocess_testa_data_fn, \
#     preprocess_train_data_fn, config_path, stopwords_fn
import config as cfg

train_data_file = cfg.original_data_path + cfg.original_train_fn
validate_data_file = cfg.original_data_path + cfg.original_validate_fn
testa_data_file = cfg.original_data_path + cfg.original_test_fn
preprocess_train_data_file = cfg.preprocess_data_path + cfg.preprocess_train_data_fn
preprocess_validate_date_file = cfg.preprocess_data_path + cfg.preprocess_validate_data_fn
preprocess_test_date_file = cfg.preprocess_data_path + cfg.preprocess_testa_data_fn
word2vec_file = cfg.word2vec_chars_path + cfg.word2vec_chars_fn
stopwords_file = cfg.config_path + cfg.stopwords_fn

stop_words = []


def parse_dataset(X_words, w2v_ind):
    """ Words become integers
    """
    data = []
    for words in X_words:
        new_txt = []
        for word in words:
            try:
                new_txt.append(w2v_ind[word])
            except KeyError:
                new_txt.append(0)
        data.append(new_txt)
    return data


def create_dictionaries(model=None, X=None):
    """创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
       Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    """
    if (X is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2v_ind = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2v_ind.keys()}  # 所有频数超过10的词语的词向量
        X = parse_dataset(X, w2v_ind)
        # 每个句子所含词语对应的索引，所有句子中含有频数小于10的词语，索引为0
        X = sequence.pad_sequences(X, maxlen=500)
        return w2v_ind, w2vec, X
    else:
        print('No data provided...')


def segWord(doc):
    seg_list = jieba.cut(doc, cut_all=False)
    return list(seg_list)


# move stop words
def filter_map(arr):
    res = ""
    for c in arr:
        if c not in stop_words and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res += c
    return res


# move stop words and generate char sent
def filter_char_map(arr):
    res = []
    for c in arr:
        if c not in stop_words and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res.append(c)
    return " ".join(res)


# get char of sentence
def get_char(arr):
    res = []
    for c in arr:
        res.append(c)
    return list(res)


def make_stopwords():
    # 生成停用词列表
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.strip('\n')
            line = line.strip('\r')
            stop_words.append(line)
    print(stop_words)
    return stop_words


cut_index = 0


def word_cut_map(text):
    global cut_index
    global stop_words
    cut_index += 1
    print('[word_cut_map %d]'%cut_index)
    line_sub = re.sub('[\s+\.\/_,$%^*();；:-【】+\'\']+|[+——！，;:。、~#￥%&*（）]+', '', text)
    word_list = jieba.lcut(line_sub)
    cut_str = ''
    for word in word_list:
        if word not in stop_words:
            cut_str += word
            cut_str += ' '
    return cut_str.split(' ')


def preprocess_text(input_file, output_file):
    texts = pd.read_csv(input_file)
    texts['content'] = texts['content'].map(lambda x: word_cut_map(x))
    texts.to_csv(output_file, index=None)
    return texts['content']


def make_word2cev_model(train_data):
    line_sent = []
    for s in train_data['content']:
        line_sent.append(s)
    word2vec_model = Word2Vec(line_sent, size=300, window=10, min_count=1, workers=4, iter=15)
    word2vec_model.wv.save_word2vec_format(word2vec_file, binary=True)


def main():
    global stop_words
    stop_words = make_stopwords()
    print('preprocess_text train_data...')
    train_data_content = preprocess_text(train_data_file, preprocess_train_data_file)
    print('preprocess_text train_data...')
    preprocess_text(validate_data_file, preprocess_validate_date_file)
    print('preprocess_text train_data...')
    preprocess_text(testa_data_file, preprocess_test_date_file)
    print('make w2v...')
    make_word2cev_model(train_data_content)


