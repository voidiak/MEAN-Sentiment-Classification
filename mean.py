import tensorflow as tf
import os
import random
import argparse
import numpy as np
from tensorpack import *
from tensorpack import LMDBSerializer, MultiProcessRunner
from tensorpack.tfutils.gradproc import GlobalNormClip, SummaryGradient
from utils import MEANBatch

WORD_EMBED_DIM = 300
EMBED_LOC = './data/wordvector/glove.refine.txt'
BATCH_SIZE = 60 

def embed(wordlist, w2v_model):
    embed_matrix = []
    for word in wordlist:
        if word in w2v_model.vocab:
            embed_matrix.append(w2v_model.word_vec(word))
        else:
            embed_matrix.append(np.zeros(WORD_EMBED_DIM))
    return np.array(embed_matrix, dtype=np.float32)

class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([None, None], tf.float32, 'input'),
                tf.TensorSpec([None, 1], tf.float32, 'label')
                ]
    
    def build_graph(self, input, label):
        return 0

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        opt = tf.train.RMSPropOptimizer(lr)
        return optimizer.apply_grad_processors(opt, [GlobalNormClip(5), SummaryGradient()])

def get_data(path, isTrain):
    ds = LMDBSerializer.load(path, shuffle=isTrain)
    ds = MEANBatch(ds, BATCH_SIZE)
    if isTrain:
        ds = MultiProcessRunner(ds, num_prefetch=2, num_proc=2)
    return ds

def get_config(ds_train, ds_dev, ds_test):
    return TrainConfig(
        data = QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_dev, ScalarStats('')),
            MaxSaver(''),
            MovingAverageSummary(),
            MergeAllSummaries(),
            GPUMemoryTracker()
        ],
        model = Model(),
        max_epoch=20
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    logger.auto_set_dir()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    tf.set_random_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    ds_train = get_data('./mdb0/train.mdb', True)
    ds_dev = get_data('./mdb0/dev.mdb', False)
    ds_test = get_data('./mdb0/test.mdb', False)
    #test dataflow
    # ds_dev.reset_state()
    # a = ds_dev.__iter__()
    # for b in a:
    #     print(b)
    config = get_config(ds_train, ds_dev, ds_test)
    if args.load:
        config.session_init = SaverRestore(args.load)
    launch_train_with_config(config, SimpleTrainer())