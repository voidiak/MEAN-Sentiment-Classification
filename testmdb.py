from tensorpack import *
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from utils import MEANBatch, word_embed

BATCH_SIZE = 60

#
# class MEANBatch(ProxyDataFlow):
#
#     def __init__(self, ds, batch):
#         self.batch = batch
#         self.ds = ds
#
#     def __len__(self):
#         return len(self.ds) // self.batch
#
#     def __iter__(self):
#         itr = self.ds.__iter__()
#         for _ in range(self.__len__()):
#             sents = []
#             labels = []
#             chars = []
#             lens = []
#             for b in range(self.batch):
#                 _chars=[]
#                 label, sentence, sent_chars, sent_len = next(itr)
#                 labels.append(label)
#                 sents.append(sentence)
#                 # chars.append(sent_chars)
#                 for i in sent_chars:
#                     _chars.append(list(i))
#                 chars.append(_chars)
#                 lens.append(sent_len)
#             max_sent_len = max(lens)
#             sents = pad_sequences(sents, max_sent_len, padding='post').tolist()
#             # sents = np.asarray(sents)
#             # chars = np.asarray(chars)
#             # labels = np.asarray(labels)
#             # lens = np.asarray(lens)
#             # max_sent_len = np.asarray([max_sent_len])
#             yield [sents, chars, labels, lens, [max_sent_len]]


def get_data(path, isTrain):
    ds = LMDBSerializer.load(path, shuffle=isTrain)
    ds = MEANBatch(ds, BATCH_SIZE)

    return ds

ds_train = get_data('./mdb0/train.mdb', True)
ds_train.__iter__()
ds_train.reset_state()
count=0
for b in ds_train:
    print('sents:{}\nchars:{}\nlabels:{}\nlens:{}\nmax_sent_len:{}\n'.format(b[0],b[1],b[2],b[3],b[4]))
    # print(b[0].shape)
    # print(b[1].shape)
    # print(b[2].shape)
    # print(b[3].shape)
    # print(b[4].shape)
    count+=1
    if count>0:
        break