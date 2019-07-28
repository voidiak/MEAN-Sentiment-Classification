import tensorflow as tf
import os
import random
import argparse
import numpy as np
from tensorpack import *
from tensorpack import LMDBSerializer, MultiProcessRunner
from tensorpack.tfutils.gradproc import GlobalNormClip, SummaryGradient
from tensorpack.tfutils import optimizer
from tensorpack.models import ConcatWith
from utils import MEANBatch, word_embed, char_embed
import six
import gensim
import pickle

WORD_EMBED_DIM = 300
CHAR_EMBED_DIM = 62 + 1
EMBED_LOC = './data/wordvector/glove.refine.txt'
BATCH_SIZE = 60 
RNN_DIM = 300
mu = 1e-4
psi = 0.9

class Model(ModelDesc):
    def __init__(self):
        self.word_vocab=pickle.load(open('./data/word_vocab.pkl', 'rb'))
        self.char_vocab=pickle.load(open('./data/char_vocab.pkl', 'rb'))
        self.regularizer=tf.contrib.layers.l2_regularizer(scale=1e-5)
        n = pickle.load(open('./data/negation.pkl', 'rb'))
        i = pickle.load(open('./data/intensifier.pkl', 'rb'))
        s = pickle.load(open('./data/sentiment.pkl', 'rb'))
        self.n_w = n['word_embedding']
        self.i_w = i['word_embedding']
        self.s_w = s['word_embedding']
        self.n_c = n['char']
        self.i_c = i['char']
        self.s_c = s['char']
        self.n_n = n['num']
        self.i_n = i['num']
        self.s_n = s['num']

    def inputs(self):
        return [tf.TensorSpec([None, None], tf.int32, 'input_w'), 
                tf.TensorSpec([None, None, None], tf.int32, 'input_c'),
                tf.TensorSpec([None, 1], tf.int32, 'label'),
                tf.TensorSpec([None, 1], tf.int32, 'x_len'),
                tf.TensorSpec((), tf.int32, 'max_len')]
    
    def build_graph(self, input_w, input_c, label, x_len, max_len):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(EMBED_LOC, binary=False)
        word_embed_init = word_embed(self.word_vocab, w2v_model)
        char_embed_init = char_embed(self.char_vocab)
        _word_embeddings = tf.get_variable('word_embeddings', initializer=word_embed_init, trainable=True,
                                            regularizer=self.regularizer)
        _char_embeddings = tf.get_variable('char_embeddings', initializer=char_embed_init, trainable=True,
                                            regularizer=self.regularizer)
        n_word = tf.get_variable('negation_word_embedding', initializer=self.n_w, trainable=True, regularizer=self.regularizer)
        i_word = tf.get_variable('intensifier_word_embedding', initializer=self.i_w, trainable=True, regularizer=self.regularizer)
        s_word = tf.get_variable('sentiment_word_embedding', initializer=self.s_w, trainable=True, regularizer=self.regularizer)
        # OOV pad
        word_zero_pad = tf.zeros([1, WORD_EMBED_DIM])
        char_zero_pad = tf.zeros([1, CHAR_EMBED_DIM])
        word_embeddings = tf.concat([word_zero_pad, _word_embeddings], axis=0)        
        char_embeddings = tf.concat([char_zero_pad, _char_embeddings], axis=0)
        word_embeded = tf.nn.embedding_lookup(word_embeddings, input_w)
        _char_embeded = tf.nn.embedding_lookup(char_embeddings, input_c)
        with tf.variable_scope('char_conv', reuse=tf.AUTO_REUSE):
            _char_embeded = Conv2D(_char_embeded, filters=300, kernel_size=(1,1), padding='same')
            _char_embeded_2 = Conv2D(_char_embeded, filters=300, kernel_size=2, padding='same')
            
            _char_embeded_3 = Conv2D(_char_embeded, filters=300, kernel_size=3, padding='same')
        char_embeded = tf.concatenate([_char_embeded_2,_char_embeded_3], axis=-1)
        context_word_repre = tf.concatenate([word_embeded, char_embeded], axis=-1)
        '''
        res  = debug_nn([de_out], self.create_feed_dict( next(self.getBatches(self.data['train'])) ) ); pdb.set_trace()
        '''
        n_c_embeded = tf.nn.embedding_lookup(char_embeddings, self.n_c)
        i_c_embeded = tf.nn.embedding_lookup(char_embeddings, self.i_c)
        s_c_embeded = tf.nn.embedding_lookup(char_embeddings, self.s_c)
        with tf.variable_scope('char_conv', reuse=tf.AUTO_REUSE):
            n_c_embeded = Conv2D(n_c_embeded, filters=300, kernel_size=(1,1), padding='same')
            n_c_embeded_2 = Conv2D(n_c_embeded, filters=300, kernel_size=2, padding='same')
            n_c_embeded_3 = Conv2D(n_c_embeded, filters=300, kernel_size=3, padding='same')
        with tf.variable_scope('char_conv', reuse=tf.AUTO_REUSE):
            i_c_embeded = Conv2D(i_c_embeded, filters=300, kernel_size=(1,1), padding='same')
            i_c_embeded_2 = Conv2D(i_c_embeded, filters=300, kernel_size=2, padding='same')
            i_c_embeded_3 = Conv2D(i_c_embeded, filters=300, kernel_size=3, padding='same')
        with tf.variable_scope('char_conv', reuse=tf.AUTO_REUSE):
            s_c_embeded = Conv2D(s_c_embeded, filters=300, kernel_size=(1,1), padding='same')
            s_c_embeded_2 = Conv2D(s_c_embeded, filters=300, kernel_size=2, padding='same')
            s_c_embeded_3 = Conv2D(s_c_embeded, filters=300, kernel_size=3, padding='same')
        negation_word_repre = ConcatWith(n_word, [n_c_embeded_2, n_c_embeded_3], -1)
        intensifier_word_repre = ConcatWith(i_word, [i_c_embeded_2, i_c_embeded_3], -1)
        sentiment_word_repre = ConcatWith(s_word, [s_c_embeded_2, s_c_embeded_3], -1)

        #TO-DO:W normalization for better calculation of word correlation

        m_s = tf.matmul(context_word_repre, sentiment_word_repre, transpose_a=True)
        m_i = tf.matmul(context_word_repre, intensifier_word_repre, transpose_a=True)
        m_n = tf.matmul(context_word_repre, negation_word_repre, transpose_a=True)

        x_c_s = tf.matmul(context_word_repre, m_s)
        x_c_i = tf.matmul(context_word_repre, m_i)
        x_c_n = tf.matmul(context_word_repre, m_n)
        x_c = tf.add(x_c_s, x_c_i, x_c_n)

        x_s = tf.matmul(sentiment_word_repre, m_s, transpose_b=True)
        x_i = tf.matmul(intensifier_word_repre, m_i, transpose_b=True)
        x_n = tf.matmul(negation_word_repre, m_n, transpose_b=True)

        gru_cell = tf.nn.rnn_cell.GRUCell(RNN_DIM, name='GRU')
        val_c, _ = tf.nn.dynamic_rnn(gru_cell, x_c, sequence_length=x_len, dtype=tf.floate32)
        val_s, _ = tf.nn.dynamic_rnn(gru_cell, x_s, sequence_length=x_len, dtype=tf.floate32)
        val_i, _ = tf.nn.dynamic_rnn(gru_cell, x_i, sequence_length=x_len, dtype=tf.floate32)
        val_n, _ = tf.nn.dynamic_rnn(gru_cell, x_n, sequence_length=x_len, dtype=tf.floate32)
        h_c = val_c[0]
        h_s = val_s[0]
        h_i = val_i[0]
        h_n = val_n[0]
        q_s = tf.reduce_mean(h_s)
        q_i = tf.reduce_mean(h_i)
        q_n = tf.reduce_mean(h_n)
        sentiment_att_weights = tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(h_c),[BATCH_SIZE, max_len, RNN_DIM]), tf.reshape(q_s, [BATCH_SIZE, RNN_DIM, 1]), [BATCH_SIZE, max_len])))

        o_1 = tf.reshape(
            tf.matmul(
                tf.reshape(sentiment_att_weights, [BATCH_SIZE, 1, max_len]),
                h_c
            ), [BATCH_SIZE, RNN_DIM]
        )
        intensifier_att_weights = tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(h_c),[BATCH_SIZE, max_len, RNN_DIM]), tf.reshape(q_i, [BATCH_SIZE, RNN_DIM, 1]), [BATCH_SIZE, max_len])))

        o_2 = tf.reshape(
            tf.matmul(
                tf.reshape(intensifier_att_weights, [BATCH_SIZE, 1, max_len]),
                h_c
            ), [BATCH_SIZE, RNN_DIM]
        )
        negation_att_weights = tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(h_c),[BATCH_SIZE, max_len, RNN_DIM]), tf.reshape(q_n, [BATCH_SIZE, RNN_DIM, 1]), [BATCH_SIZE, max_len])))

        o_3 = tf.reshape(
            tf.matmul(
                tf.reshape(negation_att_weights, [BATCH_SIZE, 1, max_len]),
                h_c
            ), [BATCH_SIZE, RNN_DIM]
        )

        output = tf.concatenate([o_1, o_2, o_3], axis=-1)

        w = tf.get_variable('w', [RNN_DIM, 2], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', initializer=np.zeros([2]).astype(np.float32))
        p_pred = tf.nn.xw_plus_b(output, w, b)
        supervised_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=p_pred, labels=label)
        l2_loss = tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #TO-DO: diversity_loss = Frobenius Norm of Matrix [o x (o)T - psi x I]



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
    config = get_config(ds_train, ds_dev, ds_test)
    if args.load:
        config.session_init = SaverRestore(args.load)
    launch_train_with_config(config, SimpleTrainer())