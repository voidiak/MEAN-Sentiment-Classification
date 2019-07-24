import unicodedata
from collections import defaultdict
import re
import os
import random
import pprint
from tensorpack.dataflow import DataFlow, LMDBSerializer
random.seed(1229)

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

def read_raw_data(data, split):
    with open("./data/mr/{}.txt".format(split), 'r', encoding='utf-8') as f:
        for line in f:
            sent = line.strip()
            sent = clean_string(sent).split(' ')
            data.append({'sent':sent, 'label':[1 if split.startswith('pos') else 0]})
    return data

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
    for id, word in enumerate(vocab):
        word2id[word] = id + 1
    return word2id

def get_word_id(word, word2id, default='NONE'):
    if word in word2id:
        return word2id[word]
    else:
        return word2id[default]

def process_data(data, word2id):
    p_data = []
    for sent in data:
        p_sent = {}
        p_sent['label'] = sent['label'] 
        p_sent['sent'] = [get_word_id(word, word2id, 'UNK') for word in sent['sent']]
        p_data.append(p_sent)
    return p_data

class MEANdata(DataFlow):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for sent in self.data:
            label=sent['label']
            sentence=sent['sent']
            yield [label, sentence]

if __name__ == "__main__":
    # read dataset
    data, intensifier, negation, sentiment= [], [], [], []
    for split in ['neg', 'pos']:
        data = read_raw_data(data, split)
    intensifier = read_resource_word(intensifier, 'intensifier')
    negation = read_resource_word(negation, 'negation')
    sentiment = read_resource_word(sentiment, 'sentiment')
    print('\nintensifier len:{}\nnegation len:{}\nsentiment:{}\n'.format(len(intensifier), len(negation), len(sentiment)))
    # make vocabulary
    vocab_stat = defaultdict(int)
    for word in (intensifier+negation+sentiment):
        vocab_stat[word] +=1
    print('resource word num:{}\n'.format(len(vocab_stat)))
    for sent in data:
        for word in sent['sent']:
            vocab_stat[word] += 1
    print('vocabulary size:{}\n'.format(len(vocab_stat)))
    freq = list(vocab_stat.items())
    freq.sort(key = lambda x: x[1], reverse=True)
    vocab, _ = map(list, zip(*freq))
    vocab.append('UNK')
    word2id = word_map(vocab)
    id2word = dict([(v,k) for k,v in word2id.items()])
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
        train_ = process_data(train, word2id)
        dev_ = process_data(dev, word2id)
        test_ = process_data(test, word2id)
        train_data = MEANdata(train_)
        dev_data = MEANdata(dev_)
        test_data = MEANdata(test_)
        os.system('mkdir mdb%s' % i)
        LMDBSerializer.save(train_data, './mdb{}/train.mdb'.format(i))
        LMDBSerializer.save(dev_data, './mdb{}/dev.mdb'.format(i))
        LMDBSerializer.save(test_data, './mdb{}/test.mdb'.format(i))
