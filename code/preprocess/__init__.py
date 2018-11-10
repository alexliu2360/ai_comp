# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import copy
import random
import re

# from gensim.corpora import Dictionary
# import keras
# from keras_preprocessing import sequence
import os
import jieba.analyse
import time

random.seed = 16
import pandas as pd
from gensim.models.word2vec import Word2Vec
import jieba
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


# def create_dictionaries(model=None, X=None):
#     """创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
#        Function does are number of Jobs:
#         1- Creates a word to index mapping
#         2- Creates a word to vector mapping
#         3- Transforms the Training and Testing Dictionaries
#     """
#     if (X is not None) and (model is not None):
#         gensim_dict = Dictionary()
#         gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
#         w2v_ind = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
#         w2vec = {word: model[word] for word in w2v_ind.keys()}  # 所有频数超过10的词语的词向量
#         X = parse_dataset(X, w2v_ind)
#         # 每个句子所含词语对应的索引，所有句子中含有频数小于10的词语，索引为0
#         X = sequence.pad_sequences(X, maxlen=500)
#         return w2v_ind, w2vec, X
#     else:
#         print('No data provided...')


def segWord(doc):
    seg_list = jieba.cut(doc, cut_all=False)
    return list(seg_list)


# move stop words
def filter_map(arr):
    # global stop_words
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


def read_stopwords():
    # 生成停用词列表
    with open(stopwords_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.strip('\n')
            line = line.strip('\r')
            stop_words.append(line)
    return stop_words


def make_stopwords(sentence):
    a = jieba.analyse.extract_tags(sentence, topK=40, withWeight=False, allowPOS=())
    print(a)
    pass


def word_cut_map(text):
    global stop_words
    # print('[word_cut_map %d]'%cut_index)
    line_sub = re.sub('[\s+\.\/_,$%^*();；:-【】+\'\']+|[+——！，;:。、~#￥%&*（）]+', '', text)
    word_list = jieba.lcut(line_sub)
    cut_str = ''
    for word in word_list:
        if word not in stop_words:
            cut_str += word
            cut_str += ' '
    return cut_str.split(' ')


def preprocess_text_jieba(input_file, output_file):
    texts = pd.read_csv(input_file)
    texts['content'] = texts['content'].map(lambda x: word_cut_map(x))
    texts.to_csv(output_file, index=None)


def preprocess_text_char(input_file, output_file):
    texts = pd.read_csv(input_file)
    texts['content'] = texts['content'].map(lambda x: filter_map(x))
    texts['content'] = texts['content'].map(lambda x: get_char(x))
    texts.to_csv(output_file, index=None)


def make_traindata_w2v_model(train_data, w2v_file):
    line_sent = []
    for s in train_data:
        line_sent.append(s)
    word2vec_model = Word2Vec(line_sent, size=100, window=10, min_count=1, workers=4, iter=15)
    word2vec_model.wv.save_word2vec_format(w2v_file, binary=True)


def strat_preprocess(arg):
    global stop_words
    stop_words = read_stopwords()
    start_t = time.time()
    if arg == 'jieba':
        if not os.path.exists(cfg.preprocess_data_path + cfg.preprocess_train_data_fn):
            print('preprocess_text_jieba train_data...')
            preprocess_text_jieba(train_data_file, preprocess_train_data_file)
        if not os.path.exists(cfg.preprocess_data_path + cfg.preprocess_validate_data_fn):
            print('preprocess_text_jieba validate_data...')
            preprocess_text_jieba(validate_data_file, preprocess_validate_date_file)
        if not os.path.exists(cfg.preprocess_data_path + cfg.preprocess_testa_data_fn):
            print('preprocess_text_jieba testa_data...')
            preprocess_text_jieba(testa_data_file, preprocess_test_date_file)
    else:
        if not os.path.exists(cfg.preprocess_data_path + cfg.preprocess_train_data_fn):
            print('preprocess_text_char train_data...')
            preprocess_text_char(train_data_file, preprocess_train_data_file)
        if not os.path.exists(cfg.preprocess_data_path + cfg.preprocess_validate_data_fn):
            print('preprocess_text_char validate_data...')
            preprocess_text_char(validate_data_file, preprocess_validate_date_file)
        if not os.path.exists(cfg.preprocess_data_path + cfg.preprocess_testa_data_fn):
            print('preprocess_text_char testa_data...')
            preprocess_text_char(testa_data_file, preprocess_test_date_file)
    print('preprocess_text time used %f s...' % (time.time() - start_t))

    # w2v_file = cfg.word2vec_chars_path + cfg.own_word2vec_fn
    w2v_file = cfg.word2vec_chars_path + cfg.own_withsw_char_word2vec_fn
    if not os.path.exists(w2v_file):
        print('start making w2v...')
        train_data = pd.read_csv(train_data_file)
        start_t = time.time()
        make_traindata_w2v_model(train_data['content'], w2v_file=w2v_file)
        print('make traindata_w2v_model time used %f s...' % (time.time() - start_t))


def read_train_data_content():
    df = pd.read_csv(cfg.original_data_path + cfg.original_train_fn)
    idx = 0
    global stop_words
    stop_words = read_stopwords()
    with open('./train_data_content_preprocessed.txt', 'w') as f:
        df['content'] = df['content'].map(lambda x: filter_map(x))
        for sentence in df['content']:
            print(idx)
            f.writelines(sentence)
            f.writelines('\n')
            idx += 1


def add_word2jieba(addwords_file):
    with open(addwords_file, 'r', encoding='utf-8') as f:
        jieba.add_word('一家人在', freq=200)
        for word in f.readlines():
            jieba.add_word(word, freq=20000)


yanwenzi_emojis_list = []


def load_yanwenzi_emojis(yanwenzi_emojis_file):
    with open(yanwenzi_emojis_file, 'r', encoding='utf8') as f:
        for s in f.readlines():
            yanwenzi_emojis_list.append(s.strip('\n'))


def is_contain_yan_emoji(s):
    global yanwenzi_emojis_list
    container = {}
    for ye in yanwenzi_emojis_list:
        index = 0
        if ye in s:
            n = s.count(ye)
            s_copy = copy.copy(s)
            for i in range(n):
                index = s_copy.find(ye)+index
                container[index] = ye
                s_copy = s_copy[index+len(ye):]
    return container


def reset_data(yanwenzi_emojis_file, test_train_file):
    global yanwenzi_emojis_list
    load_yanwenzi_emojis(yanwenzi_emojis_file)
    with open(test_train_file, 'r', encoding='utf-8') as f:
        for s in f.readlines():
            ye_container = is_contain_yan_emoji(s)
            print(ye_container)



if __name__ == '__main__':
    #     # df = pd.read_csv(cfg.original_train_fn)
    #     # for sentence in df['content']:
    #     #     make_stopwords(sentence)
    #     stop_words = read_stopwords()
    #     preprocess_text_char(cfg.test_original_train_data_10_file, 'a.txt')
    # add_word2jieba('./addwords.txt')
    reset_data('yanwenzi_emojis.txt', './test_file.txt')
