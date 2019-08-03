import unicodedata
import operator
import six
from collections import defaultdict, Counter
import re
import os
import random
import gensim
import pprint
from tensorpack.dataflow import DataFlow, LMDBSerializer
import pickle
from utils import word_embed
from keras.preprocessing.sequence import pad_sequences
import numpy as np
random.seed(1229)

EMBED_LOC = './data/wordvector/glove.refine.txt'
MAX_WORD_LEN = 20

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
     if unicodedata.category(c) != 'Mn')

def normalize_chars(w):
    if w == '-LRB-':
        return '('
    elif w == '-RRB-':
        return ')'
    elif w == '-LCB-':
        return '{'
    elif w == '-RCB-':
        return '}'
    elif w == '-LSB-':
        return '['
    elif w == '-RSB-':
        return ']'
    return w.replace(r'\/', '/').replace(r'\*', '*')

def normalize_word(w):
    return re.sub(r'\d', '0', normalize_chars(w).lower())

def clean_string(str):
    w = str
    w = unicode_to_ascii(w.strip())
    split=w.split()
    for i in range(len(split)):
        split[i]=normalize_word(split[i])
    w=' '.join(split)
    w = re.sub(r'-',' - ',w)
    w = re.sub(r'/',' / ',w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.rstrip().strip()
    return w

def read_raw_data():
    data=[]
    char_vocab=set()
    for split in ['neg','pos']:
        with open("./data/mr/{}.txt".format(split), 'r', encoding='utf-8') as f:
            for line in f:
                sent = line.strip()
                sent = clean_string(sent).split(' ')[:100]
                for word in sent:
                    for c in word:
                        char_vocab.add(c)
                data.append({'sent':sent, 'label':[1 if split.startswith('pos') else 0]})
    print(char_vocab)
    print(len(char_vocab))
    return data, char_vocab

def get_test(data, n, x):
    st, ed = len(data) * x // n, len(data) * (x+1) // n
    return data[st:ed]

def get_train(data, n, x):
    st, ed = len(data) * x // n, len(data) * (x+1) // n
    return [data[:st]] + [data[ed:]]

def read_resource_word(word_list, split):
    with open("./data/wordlist/{}.txt".format(split), 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip().split(' '))==2:
                word = line.strip().split(' ')[1]
                word_list.append(word)
            else:
                print('{} \t {}'.format(split, line))
    return word_list

def word_map(vocab):
    word2id={}
    w_max_len = 0
    for id, word in enumerate(vocab):
        word2id[word] = id + 1
        l = len(word)
        if l>w_max_len:
            w_max_len = l
    return word2id, w_max_len

def char_map(char_vocab):
    char2id={}
    for id, char in enumerate(char_vocab):
        char2id[char] = id + 1
    return char2id

def get_word_id(word, word2id, default='NONE'):
    if word in word2id:
        return word2id[word]
    else:
        return word2id[default]

def get_char_id(char, char2id, default='NONE'):
    if char in char2id:
        return char2id[char]
    else:
        return char2id[default]

def process_data(data, word2id, char2id):
    p_data = []
    for sent in data:
        p_sent = {}
        p_sent['label'] = sent['label']
        p_sent['sent'] = [get_word_id(word, word2id, 'UNK') for word in sent['sent']]
        p_sent['sent_char'] = [[get_char_id(char, char2id, 'UNK') for char in word[:MAX_WORD_LEN]] for word in sent['sent']]
        p_sent['sent_char'] = pad_sequences(p_sent['sent_char'], MAX_WORD_LEN, padding='post')
        p_sent['sent_len'] = len(p_sent['sent'])
        p_data.append(p_sent)
    return p_data

def pad_dynamic(self, x):

    temp = np.zeros((MAX_WORD_LEN), np.int32)

    for i, ele in enumerate(data):
        temp[i, :len(ele)] = ele[:MAX_WORD_LEN]

    return temp

class MEANdata(DataFlow):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sent in self.data:
            label=sent['label']
            sentence=sent['sent']
            sentence_char=sent['sent_char']
            sentence_len=sent['sent_len']
            yield [label, sentence, sentence_char, sentence_len]

if __name__ == "__main__":
    # read dataset
    intensifier, negation, sentiment= [], [], []
    data, chars = read_raw_data()
    intensifier = read_resource_word(intensifier, 'intensifier')
    negation = read_resource_word(negation, 'negation')
    sentiment = read_resource_word(sentiment, 'sentiment')
    m = len(sentiment)
    k = len(intensifier)
    p = len(negation)
    print('\nintensifier len:{}\nnegation len:{}\nsentiment:{}\n'.format(k, p, m))

    # make vocabulary
    vocab_stat = defaultdict(int)
    for word in (intensifier+negation+sentiment):
        vocab_stat[word] +=1
    print('resource word num:{}\n'.format(len(vocab_stat)))
    for sent in data:
        for word in sent['sent']:
            vocab_stat[word] += 1
    print('word vocabulary size:{}\n'.format(len(vocab_stat)))
    freq = list(vocab_stat.items())
    freq.sort(key = lambda x: x[1], reverse=True)
    vocab, _ = map(list, zip(*freq))
    vocab.append('UNK')
    word2id, w_max_len = word_map(vocab)
    print('set word max length:{}'.format(min(w_max_len, MAX_WORD_LEN)))
    chars=list(chars)
    chars.append('UNK')
    char2id = char_map(chars)
    id2word = dict([(v,k) for k,v in word2id.items()])
    id2char = dict([(v,k) for k,v in char2id.items()])
    pickle.dump(vocab, open('./data/word_vocab.pkl', 'wb'))
    pickle.dump(chars, open('./data/char_vocab.pkl', 'wb'))

    # save sentiment resource word embedding
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(EMBED_LOC, binary=False)
    w_n = word_embed(negation, w2v_model)
    w_i = word_embed(intensifier, w2v_model)
    w_s = word_embed(sentiment, w2v_model)
    # w_n = np.reshape(w_n,(-1,1,300))
    # w_i = np.reshape(w_i,(-1,1,300))
    # w_s = np.reshape(w_s,(-1,1,300))
    negation_c = [[get_char_id(char, char2id, 'UNK') for char in word[:MAX_WORD_LEN]] for word in negation]
    intensifier_c = [[get_char_id(char, char2id, 'UNK') for char in word[:MAX_WORD_LEN]] for word in intensifier]
    sentiment_c = [[get_char_id(char, char2id, 'UNK') for char in word[:MAX_WORD_LEN]] for word in sentiment]
    negation_c = pad_sequences(negation_c, MAX_WORD_LEN, padding='post')
    intensifier_c = pad_sequences(intensifier_c, MAX_WORD_LEN, padding='post')
    sentiment_c = pad_sequences(sentiment_c, MAX_WORD_LEN, padding='post')

    pickle.dump({'word_embedding':w_i, 'char':intensifier_c, 'num':k}, open('./data/intensifier.pkl', 'wb'))
    pickle.dump({'word_embedding':w_n, 'char':negation_c, 'num':p}, open('./data/negation.pkl', 'wb'))
    pickle.dump({'word_embedding':w_s, 'char':sentiment_c, 'num':m}, open('./data/sentiment.pkl', 'wb'))
    # make train/dev/test
    for i in range(10):
        train_ori = get_train(data, 10, i)
        test_ori = get_test(data, 10, i)

        train = []
        dev = []
        test = []
        for j in range(2):
            random.shuffle(train_ori[j])
            x = len(train_ori[j]) * 9 // 10
            train += train_ori[j][:x]
            dev += train_ori[j][x:]
        test += test_ori
        random.shuffle(train)
        random.shuffle(dev)
        random.shuffle(test)
        train_ = process_data(train, word2id, char2id)
        dev_ = process_data(dev, word2id, char2id)
        test_ = process_data(test, word2id, char2id)
        train_data = MEANdata(train_)
        dev_data = MEANdata(dev_)
        test_data = MEANdata(test_)
        os.system('mkdir mdb%s' % i)
        LMDBSerializer.save(train_data, './mdb{}/train.mdb'.format(i))
        LMDBSerializer.save(dev_data, './mdb{}/dev.mdb'.format(i))
        LMDBSerializer.save(test_data, './mdb{}/test.mdb'.format(i))
