from tensorpack import ProxyDataFlow
from six.moves import range
import numpy as np

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
            for b in range(self.batch):
                label, sentence = next(itr)
                labels.append(label)
                sents.append(sentence)
            yield [sents, labels]