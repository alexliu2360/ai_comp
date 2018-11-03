# -*- coding: utf-8 -*-
import random

random.seed = 16
import pandas as pd
from gensim.models.word2vec import Word2Vec
import jieba

data_path = '/home/competition/alexliu/BiGRU/data/'
preprocess_path = "preprocess/"
word2vec = "word2vec/"
train_data_filename = "sentiment_analysis_trainingset.csv"
validate_data_filename = "sentiment_analysis_validationset.csv"
testa_data_filename = "sentiment_analysis_testa.csv"

data = pd.read_csv(data_path + train_data_filename)

stopwords = []
with open("stopwords.txt") as f:
    for line in f.readlines():
        line = line.strip()
        stopwords.append(line)


def segWord(doc):
    seg_list = jieba.cut(doc, cut_all=False)
    return list(seg_list)


# move stop words
def filter_map(arr):
    res = ""
    for c in arr:
        if c not in stopwords and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res += c
    return res


# move stop words and generate char sent
def filter_char_map(arr):
    res = []
    for c in arr:
        if c not in stopwords and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res.append(c)
    return " ".join(res)


# get char of sentence
def get_char(arr):
    res = []
    for c in arr:
        res.append(c)
    return list(res)


data.content = data.content.map(lambda x: filter_map(x))
data.content = data.content.map(lambda x: get_char(x))

data.to_csv(preprocess_path + "train_char.csv", index=None)

line_sent = []
for s in data["content"]:
    line_sent.append(s)
word2vec_model = Word2Vec(line_sent, size=100, window=10, min_count=1, workers=4, iter=15)
word2vec_model.wv.save_word2vec_format(word2vec+"chars.vector", binary=True)

validation = pd.read_csv(data_path + validate_data_filename)
validation.content = validation.content.map(lambda x: filter_map(x))
validation.content = validation.content.map(lambda x: get_char(x))

validation.to_csv(preprocess_path + "validation_char.csv", index=None)
test = pd.read_csv(data_path + testa_data_filename)

test.content = test.content.map(lambda x: filter_map(x))
test.content = test.content.map(lambda x: get_char(x))
test.to_csv(preprocess_path + "test_char.csv", index=None)
