from tensorpack import ProxyDataFlow
from six.moves import range
import numpy as np
from keras.preprocessing.sequence import pad_sequences
WORD_EMBED_DIM = 300
CHAR_EMBED_DIM = 62 + 1

def word_embed(wordlist, w2v_model):
    embed_matrix = []
    for word in wordlist:
        if word in w2v_model.vocab:
            embed_matrix.append(w2v_model.word_vec(word))
        else:
            embed_matrix.append(np.zeros(WORD_EMBED_DIM))
    return np.array(embed_matrix, dtype=np.float32)

def char_embed(charlist):
    char_matrix =[]
    char_vocab = np.eye(CHAR_EMBED_DIM)
    for char in charlist:
        if char in char_vocab:
            char_matrix.append(char_vocab[char])
        else:
            char_matrix.append(np.zeros(CHAR_EMBED_DIM))
    return np.array(char_matrix, dtype=np.float32)

class MEANBatch(ProxyDataFlow):

    def __init__(self, ds, batch):
        self.batch = batch
        self.ds = ds
    
    def __len__(self):
        return len(self.ds) // self.batch
    
    def __iter__(self):
        itr = self.ds.__iter__()
        for _ in range(self.__len__()):
            sents=[]
            labels=[]
            chars=[]
            lens=[]
            for b in range(self.batch):
                label, sentence, sent_chars, sent_len = next(itr)
                # TODO:PAD Sentences
                labels.append(label)
                sents.append(sentence)
                chars.append(sent_chars)
                lens.append(sent_len)
            max_sent_len = max(lens)
            sents = pad_sequences(sents, max_sent_len, padding='post')
            yield [sents, chars, labels, lens, max_sent_len]