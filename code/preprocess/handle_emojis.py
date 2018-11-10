# -*- coding: utf-8 -*-
import pandas as pd
from datashape import unicode
from emoji.core import emoji_lis
import config as cfg

stop_words = []


def is_chinese(uchar):
    # if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
    if uchar >= u'\u4e00' and uchar <= u'\u9fff':
        return True
    else:
        return False


def read_stopwords():
    # 生成停用词列表
    with open('./stopwords.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.strip('\n')
            line = line.strip('\r')
            stop_words.append(line)
    return stop_words


def not_chinese(uchar):
    # if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
    if uchar <= u'\u4e00' or uchar >= u'\u9fff':
        return True
    else:
        return False


def is_number(uchar):
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


# move stop words
def filter_map(arr):
    res = ""
    # arr = str(arr, encoding='utf8')
    for c in arr:
        if is_chinese(c) and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res += c
    return res


# move stop words
def filter_map_not_chinese(arr):
    res = ""
    # arr = str(arr, encoding='utf8')
    for c in arr:
        if c not in stop_words and not_chinese(
                c) and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res += c
    return res


def format_str(content):
    content = unicode(content, 'utf-8')
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str


def read_train_data_content():
    df = pd.read_csv(cfg.original_data_path + cfg.original_train_fn)
    global stop_words
    stop_words = read_stopwords()
    idx = 0
    with open('filter_sw.txt', 'w', encoding='utf-8') as new_f:
        df['content'] = df['content'].map(lambda x: filter_map_not_chinese(x))
        for sentence in df['content']:
            sentence = filter_map_not_chinese(sentence)
            if sentence:
                new_f.writelines(sentence)
                new_f.writelines('\n')
            print(idx)
            idx += 1


def read_all_emojis_from_trained_data(emojis_file):
    df = pd.read_csv(cfg.original_data_path + cfg.original_train_fn)
    emojis_list = []
    with open(emojis_file, 'w', encoding='utf-8') as f:
        for sentence in df['content']:
            e = emoji_lis(sentence)
            emojis_list.extend(e)
        for e in emojis_list:
            f.write(e['emoji'])


def clean_redundant_emojis(utreated_emojis_file, solved_emojis_file):
    emojis_list = []
    with open(utreated_emojis_file, 'r', encoding='utf8') as uef:
        for sentence in uef.readlines():
            for s in sentence:
                if s not in emojis_list:
                    emojis_list.append(s)
        with open(solved_emojis_file, 'w') as sef:
            for e in emojis_list:
                sef.write(e)
                sef.write('\n')


def clean_redundant_yanwenzi(utreated_yanwenzi_file, solved_yanwenzi_file):
    yanwenzi_list = []
    with open(utreated_yanwenzi_file, 'r', encoding='utf8') as uef:
        for sentence in uef.readlines():
            if sentence not in yanwenzi_list:
                yanwenzi_list.append(sentence)
        with open(solved_yanwenzi_file, 'w') as sef:
            for e in yanwenzi_list:
                sef.write(e)


def add_frep(thefile, newfile, freq=2000):
    new_s_list = []
    with open(thefile, 'r', encoding='utf8') as f:
        for s in f.readlines():
            new_s = s.strip('\n') + ' ' + str(freq) + '\n'
            new_s_list.append(new_s)
        with open(newfile, 'w', encoding='utf8') as new_f:
            for s in new_s_list:
                new_f.write(s)


if __name__ == '__main__':
    # read_train_data_content()
    # read_all_emojis_from_trained_data('./unsolved_emojis.txt')
    # clean_redundant_emojis('./unsolved_emojis.txt', './solved_emojis.txt')
    # clean_redundant_yanwenzi('./yanwenzi.txt', './sovled_yanwenzi.txt')
    add_frep('./yanwenzi_emojis.txt', './user_dict.txt')
    # with open('./emoji_test.txt', 'r', encoding='utf-8') as f:
    #     for s in f.readlines():
    #         filter_map_not_chinese(s)
    #         print(len(s))
    # for i in range(0x1f600, 0x1f650):
    #     print(chr(i), end=" ")
    #     if i % 16 == 15:
    #         print()
    # for i in range(0x0, 0x10ffff):
    #     print(chr(i), end=" ")
    #     if i % 16 == 15:
    #         print()
