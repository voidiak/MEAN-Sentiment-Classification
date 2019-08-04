import tensorflow as tf
import os
import random
import argparse
import numpy as np
from tensorpack import *
from tensorpack import LMDBSerializer, MultiProcessRunner
from tensorpack.tfutils.gradproc import GlobalNormClip, SummaryGradient
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.models import ConcatWith, BatchNorm
from utils import MEANBatch, word_embed
import six
import gensim
import pickle

WORD_EMBED_DIM = 300
CHAR_EMBED_DIM = 56 + 1
EMBED_LOC = './data/wordvector/glove.refine.txt'
BATCH_SIZE = 60
RNN_DIM = 300
mu = 1e-4
psi = 0.9
MAX_WORD_LEN = 20
MAX_LEN = 100
EPOCHS = 40


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

    def inputs(self):
        return [tf.TensorSpec([None, None], tf.int32, 'input_w'),
                tf.TensorSpec([None, None, None], tf.int32, 'input_c'),
                tf.TensorSpec([None, None], tf.int32, 'label'),
                tf.TensorSpec([None], tf.int32, 'x_len'),
                tf.TensorSpec((), tf.int32, 'max_len')]

    def build_graph(self, input_w, input_c, label, x_len, max_len):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(EMBED_LOC, binary=False)
        word_embed_init = word_embed(self.word_vocab, w2v_model)
        char_embed_init = tf.eye(CHAR_EMBED_DIM, dtype=tf.float32)
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
        _char_embeded = tf.reshape(_char_embeded, [BATCH_SIZE, max_len, MAX_WORD_LEN, CHAR_EMBED_DIM])

        negation_c = tf.get_variable('negation_char', initializer=self.n_c, trainable=False, dtype=tf.int32)
        intensifier_c = tf.get_variable('intensifier_char', initializer=self.i_c, trainable=False, dtype=tf.int32)
        sentiment_c = tf.get_variable('sentiment_char', initializer=self.s_c, trainable=False, dtype=tf.int32)

        n_c_embeded = tf.nn.embedding_lookup(char_embeddings, negation_c)
        i_c_embeded = tf.nn.embedding_lookup(char_embeddings, intensifier_c)
        s_c_embeded = tf.nn.embedding_lookup(char_embeddings, sentiment_c)

        n_c_embeded = tf.expand_dims(n_c_embeded, axis=0)
        i_c_embeded = tf.expand_dims(i_c_embeded, axis=0)
        s_c_embeded = tf.expand_dims(s_c_embeded, axis=0)

        @auto_reuse_variable_scope
        def conv1(x):
            return Conv2D('conv1x1', x, 1, (1,1), padding='same')

        @auto_reuse_variable_scope
        def conv2(x):
            return Conv2D('conv2gram', x, 150, (2, 20), strides=(1, 20), padding='same')

        @auto_reuse_variable_scope
        def conv3(x):
            return Conv2D('conv3gram', x, 150, (3, 20), strides=(1, 20), padding='same')

        _char = conv1(_char_embeded)
        _char_2 = conv2(_char)
        _char_3 = conv3(_char)
        char_embeded = tf.concat([_char_2,_char_3], axis=-1)
        char_embeded = tf.squeeze(char_embeded, [2])
        context_word_repre = tf.concat([word_embeded, char_embeded], axis=-1)

        n_c_embeded = conv1(n_c_embeded)
        n_char_2 = conv2(n_c_embeded)
        n_char_3 = conv3(n_c_embeded)
        n_char_embeded = tf.concat([n_char_2, n_char_3], axis=-1)
        n_char_embeded = tf.squeeze(n_char_embeded, [2])
        negation_word_repre = tf.concat([tf.expand_dims(n_word, 0), n_char_embeded], axis=-1)

        i_c_embeded = conv1(i_c_embeded)
        i_char_2 = conv2(i_c_embeded)
        i_char_3 = conv3(i_c_embeded)
        i_char_embeded = tf.concat([i_char_2, i_char_3], axis=-1)
        i_char_embeded = tf.squeeze(i_char_embeded, [2])
        intensifier_word_repre = tf.concat([tf.expand_dims(i_word, 0), i_char_embeded], axis=-1)

        s_c_embeded = conv1(s_c_embeded)
        s_char_2 = conv2(s_c_embeded)
        s_char_3 = conv3(s_c_embeded)
        s_char_embeded = tf.concat([s_char_2, s_char_3], axis=-1)
        s_char_embeded = tf.squeeze(s_char_embeded, [2])
        sentiment_word_repre = tf.concat([tf.expand_dims(s_word, 0), s_char_embeded], axis=-1)

        context_word_repre = tf.expand_dims(context_word_repre, -1)
        negation_word_repre = tf.expand_dims(negation_word_repre, -1)
        intensifier_word_repre = tf.expand_dims(intensifier_word_repre, -1)
        sentiment_word_repre = tf.expand_dims(sentiment_word_repre, -1)

        context_word_repre = BatchNorm('norm_c',context_word_repre, axis=3)
        negation_word_repre = BatchNorm('norm_n',negation_word_repre, axis=3)
        intensifier_word_repre = BatchNorm('norm_i',intensifier_word_repre, axis=3)
        sentiment_word_repre = BatchNorm('norm_s',sentiment_word_repre, axis=3)

        negation_word_repre = tf.squeeze(tf.tile(negation_word_repre, [BATCH_SIZE, 1, 1, 1]), [3])
        intensifier_word_repre = tf.squeeze(tf.tile(intensifier_word_repre, [BATCH_SIZE, 1, 1, 1]), [3])
        sentiment_word_repre = tf.squeeze(tf.tile(sentiment_word_repre, [BATCH_SIZE, 1, 1, 1]), [3])
        context_word_repre = tf.squeeze(context_word_repre, [3])

        m_s = tf.matmul(context_word_repre, sentiment_word_repre, transpose_b=True)
        m_i = tf.matmul(context_word_repre, intensifier_word_repre, transpose_b=True)
        m_n = tf.matmul(context_word_repre, negation_word_repre, transpose_b=True)

        x_s = tf.matmul(context_word_repre, m_s, transpose_a=True)
        x_i = tf.matmul(context_word_repre, m_i, transpose_a=True)
        x_n = tf.matmul(context_word_repre, m_n, transpose_a=True)

        x_c_s = tf.matmul(sentiment_word_repre, m_s, transpose_a=True, transpose_b=True)
        x_c_i = tf.matmul(intensifier_word_repre, m_i, transpose_a=True, transpose_b=True)
        x_c_n = tf.matmul(negation_word_repre, m_n, transpose_a=True, transpose_b=True)
        x_c = x_c_s + x_c_i + x_c_n

        x_c = tf.reshape(x_c, [BATCH_SIZE, -1, 600])
        x_s = tf.reshape(x_s, [BATCH_SIZE, -1, 600])
        x_i = tf.reshape(x_i, [BATCH_SIZE, -1, 600])
        x_n = tf.reshape(x_n, [BATCH_SIZE, -1, 600])

        # h_c = tf.keras.layers.CuDNNGRU(RNN_DIM, return_sequences=True, name='gru_c')(x_c)
        # h_s = tf.keras.layers.CuDNNGRU(RNN_DIM, return_sequences=True, name='gru_s')(x_s)
        # h_i = tf.keras.layers.CuDNNGRU(RNN_DIM, return_sequences=True, name='gru_i')(x_i)
        # h_n = tf.keras.layers.CuDNNGRU(RNN_DIM, return_sequences=True, name='gru_n')(x_n)
        c_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(RNN_DIM, name='c_GRU'), output_keep_prob=0.5)
        s_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(RNN_DIM, name='s_GRU'), output_keep_prob=0.5)
        i_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(RNN_DIM, name='i_GRU'), output_keep_prob=0.5)
        n_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(RNN_DIM, name='n_GRU'), output_keep_prob=0.5)
        h_c, _ = tf.nn.dynamic_rnn(c_cell, x_c, sequence_length=x_len, dtype=tf.float32)
        h_s, _ = tf.nn.dynamic_rnn(s_cell, x_s, sequence_length=x_len, dtype=tf.float32)
        h_i, _ = tf.nn.dynamic_rnn(i_cell, x_i, sequence_length=x_len, dtype=tf.float32)
        h_n, _ = tf.nn.dynamic_rnn(n_cell, x_n, sequence_length=x_len, dtype=tf.float32)

        q_s = tf.reduce_mean(h_s, 1, keepdims=True)
        q_i = tf.reduce_mean(h_i, 1, keepdims=True)
        q_n = tf.reduce_mean(h_n, 1, keepdims=True)
        sentiment_att_weights = tf.nn.softmax(
            tf.reshape(tf.matmul(tf.tanh(h_c), tf.reshape(q_s, [BATCH_SIZE, RNN_DIM, 1])), [BATCH_SIZE, max_len]))

        o_1 = tf.reshape(
            tf.matmul(
                tf.reshape(sentiment_att_weights, [BATCH_SIZE, 1, max_len]),
                h_c
            ), [BATCH_SIZE, RNN_DIM]
        )
        intensifier_att_weights = tf.nn.softmax(
            tf.reshape(tf.matmul(tf.tanh(h_c), tf.reshape(q_i, [BATCH_SIZE, RNN_DIM, 1])), [BATCH_SIZE, max_len]))

        o_2 = tf.reshape(
            tf.matmul(
                tf.reshape(intensifier_att_weights, [BATCH_SIZE, 1, max_len]),
                h_c
            ), [BATCH_SIZE, RNN_DIM]
        )
        negation_att_weights = tf.nn.softmax(
            tf.reshape(tf.matmul(tf.tanh(h_c), tf.reshape(q_n, [BATCH_SIZE, RNN_DIM, 1])), [BATCH_SIZE, max_len]))

        o_3 = tf.reshape(
            tf.matmul(
                tf.reshape(negation_att_weights, [BATCH_SIZE, 1, max_len]),
                h_c
            ), [BATCH_SIZE, RNN_DIM]
        )

        output = tf.concat([o_1, o_2, o_3], axis=-1)

        w = tf.get_variable('w', [3*RNN_DIM, 2], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', initializer=np.zeros([2]).astype(np.float32))
        p_pred = tf.nn.xw_plus_b(output, w, b)
        p_pred = Dropout(p_pred, keep_prob=0.5)
        labels = tf.one_hot(label, 2, axis=-1, dtype=tf.int32, name='onehot_label')
        supervised_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=p_pred, labels=labels))
        l2_loss = tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        diversity_loss = tf.norm(tf.matmul(output, output, transpose_b=True)-psi*tf.eye(BATCH_SIZE), ord='fro', axis=[-2, -1])
        loss = supervised_loss + l2_loss + mu*diversity_loss
        loss = tf.identity(loss, "total_loss")
        y_pred = tf.argmax(tf.nn.softmax(p_pred), axis=1, output_type=tf.int32)
        label_ = tf.reshape(label, [BATCH_SIZE, ])
        accuracy_ = tf.cast(tf.equal(y_pred, label_), tf.float32, name='accu')
        mean_accuracy = tf.reduce_mean(accuracy_)
        summary.add_moving_summary(loss, mean_accuracy)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=5e-3, trainable=False)
        opt = tf.train.RMSPropOptimizer(lr)
        return opt

def get_data(path, isTrain):
    ds = LMDBSerializer.load(path, shuffle=isTrain)
    ds = MEANBatch(ds, BATCH_SIZE)
    if isTrain:
        ds = MultiProcessRunnerZMQ(ds, 4)
    return ds

def get_config(ds_train, ds_dev, ds_test):
    return TrainConfig(
        data = QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_dev, [ScalarStats('total_loss'), ClassificationError('accu', 'accuracy')]),
            MaxSaver('accuracy'),
            MovingAverageSummary(),
            MergeAllSummaries(),
            GPUMemoryTracker()
        ],
        # steps_per_epoch=100,
        model = Model(),
        max_epoch=EPOCHS
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    logger.auto_set_dir(action='d')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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