import tensorflow as tf
import numpy as np

import keras
import keras.backend.tensorflow_backend as KTF
from keras.layers import *
from keras.layers.core import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.activations import softmax
from keras.models import load_model
from keras.models import Model
from keras.callbacks import Callback
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from keras import regularizers

from sklearn.model_selection import KFold

import itertools
import logging
from time import time
import os
import pickle
import sys
import datetime
import json
import pandas as pd
import numpy as np

from common_module import Attention

import data_helper as dh
from keras.optimizers import *


def unchanged_shape(input_shape):
    return input_shape


def substract(input_1, input_2):
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])

    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    # 这里相当于矩阵转置了
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))

    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


class AttentivePoolingLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentivePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 定义需要学习的参数
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        feature_dim = input_shape[-1]
        self.U = self.add_weight(shape=(feature_dim, feature_dim), name='U', initializer='uniform', trainable=True)
        self.built = True
        super(AttentivePoolingLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        feature_dim = input_shape[0][-1]
        return [(input_shape[0][0], feature_dim),(input_shape[1][0], feature_dim)]
        # return input_shape

    def call(self, inputs, **kwargs):
        q = inputs[0]
        a = inputs[1]

        time_step = inputs[0].shape[1].value
        feature_dim = inputs[0].shape[2].value

        G = tf.tensordot(q, self.U, axes=1)
        G = tf.reshape(G, shape=(-1, time_step, feature_dim), )
        G = tf.matmul(G, a, transpose_b=True)
        G = tf.nn.tanh(G)

        g_q = tf.reduce_max(G, axis=-1)
        g_a = tf.reduce_max(G, axis=1)

        g_q = tf.expand_dims(g_q, axis=-1)
        g_a = tf.expand_dims(g_a, axis=-1)

        g_q = tf.nn.softmax(g_q, dim=1)
        g_a = tf.nn.softmax(g_a, dim=1)

        # q = q * g_q
        # a = a * g_a

        q = tf.matmul(q, g_q, transpose_a=True)
        q = tf.squeeze(q, [-1])
        a = tf.matmul(a, g_a, transpose_a=True)
        a = tf.squeeze(a, [-1])

        return [q, a]


def esim(embeddings, gl_embed_matrix, maxlen, lstm_dim=256, spatial_dropout=0.3, dense_dropout=0.5, lr=0.001, f_dim=0):
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    q1_gl = Input(name='q1_gl', shape=(maxlen,))
    q2_gl = Input(name='q2_gl', shape=(maxlen,))

    f_input = Input(shape=(f_dim,))

    embedding = Embedding(embeddings.shape[0], embeddings.shape[-1], weights=[embeddings], input_length=maxlen,
                          trainable=False)
    gl_embedding = Embedding(gl_embed_matrix.shape[0], gl_embed_matrix.shape[1], weights=[gl_embed_matrix],
                             input_length=maxlen, trainable=False)

    q1_embed = BatchNormalization(axis=-1)(embedding(q1))
    q2_embed = BatchNormalization(axis=-1)(embedding(q2))

    q1_embed_gl = BatchNormalization()(gl_embedding(q1_gl))
    q2_embed_gl = BatchNormalization()(gl_embedding(q2_gl))

    q1_embed = Concatenate()([q1_embed, q1_embed_gl])
    q2_embed = Concatenate()([q2_embed, q2_embed_gl])

    # q_embed_sub = Subtract()([q1_embed, q2_embed])
    # q_embed_mul = Multiply()([q1_embed, q2_embed])
    #
    # q_embed_rep = Concatenate()([GlobalMaxPool1D()(q_embed_sub), GlobalMaxPool1D()(q_embed_mul)])

    # q1_embed = SpatialDropout1D(0.3)(q1_embed)
    # q2_embed = SpatialDropout1D(0.3)(q2_embed)

    # Encode
    encode = Bidirectional(CuDNNGRU(256, return_sequences=True))
    # encode2 = Bidirectional(CuDNNGRU(128, return_sequences=True))

    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # q1_embed_enhance = Concatenate()([q1_encoded, q1_embed_gl])
    # q2_embed_enhance = Concatenate()([q2_encoded, q2_embed_gl])

    # q1_encoded = encode2(q1_embed_enhance)
    # q2_encoded = encode2(q2_embed_enhance)

    # convolution layer

    conv_filter_size = 6
    conv = Conv1D(filters=128, kernel_size=2, activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')
    q1_conv = conv(q1_embed)
    q1_conv = Dropout(0.3)(q1_conv)
    q1_conv = conv2(q1_conv)

    q2_conv = conv(q2_embed)
    q2_conv = Dropout(0.3)(q2_conv)
    q2_conv = conv2(q2_conv)

    q1_conv_rep, q2_conv_rep = AttentivePoolingLayer()([q1_conv, q2_conv])

    q_conv_rep = Concatenate()([Subtract()([q1_conv_rep, q2_conv_rep]), Multiply()([q1_conv_rep, q2_conv_rep])])
    q_conv_rep = BatchNormalization()(q_conv_rep)
    q_conv_rep = Dropout(0.5)(q_conv_rep)

    # global maxpooling
    combine_list = []
    combine_list.append(Subtract()([GlobalMaxPool1D()(q1_encoded), GlobalMaxPool1D()(q2_encoded)]))
    combine_list.append(Multiply()([GlobalMaxPool1D()(q1_encoded), GlobalMaxPool1D()(q2_encoded)]))

    q_embed_rep = Concatenate()(combine_list)
    q_embed_rep = BatchNormalization()(q_embed_rep)
    q_embed_rep = Dropout(0.5)(q_embed_rep)

    # att_layer = Attention(maxlen)
    #
    # q1_att = att_layer(q1_encoded)
    # q2_att = att_layer(q2_encoded)
    #
    # sub_att = Subtract()([q1_att, q2_att])
    # mul_att = Multiply()([q1_att, q2_att])
    #
    # att_rep = Concatenate(axis=-1)([q1_att, q2_att, sub_att, mul_att])
    # att_rep = BatchNormalization(axis=-1)(att_rep)
    # att_rep = Dropout(0.5)(att_rep)

    q1_encoded = SpatialDropout1D(spatial_dropout)(q1_encoded)
    q2_encoded = SpatialDropout1D(spatial_dropout)(q2_encoded)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    # 卷积
    # filter_size_list = [2, 3, 4, 5]
    # result = []
    # for filter_size in filter_size_list:
    #     conv = Conv1D(filters=256, kernel_size=filter_size)
    #     q1_conv = conv(q1_combined)
    #     q2_conv = conv(q2_combined)
    #     result.append(Concatenate()([Subtract()([GlobalMaxPool1D()(q1_conv), GlobalMaxPool1D()(q2_conv)]),
    #                                Multiply()([GlobalMaxPool1D()(q1_conv), GlobalMaxPool1D()(q2_conv)])]))
    # conv_result = Concatenate()(result)
    # # conv_result = Dense(units=256, activation='relu')(conv_result)
    # conv_result = Dropout(0.5)(conv_result)
    # q1_combined_rep = GlobalMaxPool1D()(q1_combined)
    # q2_combined_rep = GlobalMaxPool1D()(q2_combined)
    #
    # q_combined_rep = Concatenate()([q1_combined_rep, q2_combined_rep])
    # q_combined_rep = Dropout(0.5)(q_combined_rep)

    q1_combined = BatchNormalization(axis=-1)(q1_combined)
    q2_combined = BatchNormalization(axis=-1)(q2_combined)

    compose = Bidirectional(CuDNNGRU(lstm_dim, return_sequences=True))

    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)

    q1_compare = SpatialDropout1D(spatial_dropout)(q1_compare)
    q2_compare = SpatialDropout1D(spatial_dropout)(q2_compare)

    # q1_ap, q2_ap = soft_attention_alignment(q1_compare, q2_compare)
    #
    # q1_combined = Concatenate()([q1_compare, q2_ap, submult(q1_compare, q2_ap)])
    # q2_combined = Concatenate()([q2_compare, q1_ap, submult(q2_compare, q1_ap)])
    #
    # q1_combined = BatchNormalization()(q1_combined)
    # q2_combined = BatchNormalization()(q2_combined)
    #
    # q1_combined = SpatialDropout1D(0.25)(q1_combined)
    # q2_combined = SpatialDropout1D(0.25)(q2_combined)
    #
    # re_encoder = Bidirectional(CuDNNGRU(lstm_dim, return_sequences=True))
    #
    # q1_compare = re_encoder(q1_combined)
    # q2_compare = re_encoder(q2_combined)

    # q1_rep = apply_multiple(q1_compare, [GlobalMaxPool1D()])
    # q2_rep = apply_multiple(q2_compare, [GlobalMaxPool1D()])

    q1_rep = GlobalMaxPool1D()(q1_compare)
    q2_rep = GlobalMaxPool1D()(q2_compare)
    # q1_rep, q2_rep = AttentivePoolingLayer()([q1_compare, q2_compare])

    q_rep = Concatenate()([q1_rep, q2_rep])
    # q_rep = BatchNormalization(axis=-1)(q_rep)
    # q_rep = Dropout(0.2)(q_rep)

    # Classifier
    merged = Concatenate()([q_rep,  q_embed_rep, q_conv_rep, f_input])
    # merged = conv_result

    print(merged)

    dense = BatchNormalization()(merged)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(512, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2, q1_gl, q2_gl, f_input], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def data_pipeline(file_name, vocabulary, maxlen=15, return_label=False):
    questions = pd.read_csv('../data/question_id.csv')
    data = pd.read_csv(file_name)
    data.rename(columns={'qid1': 'qid2', 'qid2': 'qid1'}, inplace=True)

    tmp = pd.merge(data, questions, left_on='qid1', right_on='qid', how='left')
    q1 = list(tmp['cid'].values)

    tmp = pd.merge(data, questions, left_on='qid2', right_on='qid', how='left')
    q2 = list(tmp['cid'].values)

    q1_list, q2_list = [], []
    for idx in range(len(q1)):
        words = q1[idx].split()
        new_words = []
        for w in words:
            if w in vocabulary:
                new_words.append(vocabulary[w])
            else:
                new_words.append(vocabulary['<unk>'])
        q1_list.append(new_words)

    for idx in range(len(q2)):
        words = q2[idx].split()
        new_words = []
        for w in words:
            if w in vocabulary:
                new_words.append(vocabulary[w])
            else:
                new_words.append(vocabulary['<unk>'])
        q2_list.append(new_words)

    q1_list = pad_sequences(q1_list, maxlen=maxlen, padding='post', truncating='post')
    q2_list = pad_sequences(q2_list, maxlen=maxlen, padding='post', truncating='post')

    if return_label is True:
        y = data['label'].values
        return q1_list, q2_list, y
    else:
        return q1_list, q2_list


config = {
    'maxlen': 25,
    'batch_size': 256,
    'epochs': 50,
    'log_file': '../log/esim_char_f1.csv',
    'result_file': '../submit/esim_char_v2.csv',
    'model_path': '../models/esim_char/kf_{}.h5'
}


class CB(Callback):
    def __init__(self, val_data, tolerates=10, model_path='../models/tmp.h5'):
        self.val_data = val_data
        self.max_f1 = 0.0
        self.tolerates = tolerates
        self.current_tolerate = 0
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):

        y_true = self.val_data[-1]
        y_pred = self.model.predict(self.val_data[0])
        y_pred = np.where(y_pred > 0.5, 1, 0)

        metrics = {}
        metrics['f1'] = f1_score(y_true, y_pred)
        print()
        print('-val_f1:{:.4f} - max_f1:{:.4f}'.format(metrics['f1'], self.max_f1))
        print()
        if metrics['f1'] > self.max_f1:
            self.max_f1 = metrics['f1']
            self.model.save_weights(self.model_path)
            self.current_tolerate = 0
        else:
            self.current_tolerate += 1

        if self.current_tolerate >= self.tolerates:
            self.model.stop_training = True

    def get_max_f1(self):
        return self.max_f1


def train():
    # 设置内存自适应
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    KTF.set_session(session)

    vocabulary, embed_matrix = dh.load_embed(file_name='../data/char_embedding.txt')
    gl_vocabulary, gl_embed_matrix = dh.load_embed(file_name='../data/gl_char_vectors.txt')

    train_file_name = '../data/train.csv'
    test_file_name = '../data/test.csv'
    test_data = pd.read_csv(test_file_name)
    q1, q2, y = data_pipeline(train_file_name, vocabulary=vocabulary, return_label=True, maxlen=config['maxlen'])
    q1_gl, q2_gl = data_pipeline(train_file_name, vocabulary=gl_vocabulary, return_label=False, maxlen=config['maxlen'])

    q1_test, q2_test = data_pipeline(test_file_name, vocabulary=vocabulary, return_label=False, maxlen=config['maxlen'])

    train_feature = pd.read_csv('../feature/train_feature.csv')
    train_feature = train_feature.values
    test_feature = pd.read_csv('../feature/test_feature.csv')
    test_feature = test_feature.values

    print('Feature dim : {}'.format(test_feature.shape[-1]))

    kf = KFold(n_splits=5)
    split_id = 0
    f1_record = []
    for t_idx, v_idx in kf.split(q1, q2, y):
        split_id += 1

        print('Kfold [{}]'.format(split_id))

        q1_train = q1[t_idx]
        q1_val = q1[v_idx]

        q2_train = q2[t_idx]
        q2_val = q2[v_idx]

        q1_gl_train = q1_gl[t_idx]
        q1_gl_val = q1_gl[v_idx]

        q2_gl_train = q2_gl[t_idx]
        q2_gl_val = q2_gl[v_idx]

        f_train = train_feature[t_idx]
        f_val = train_feature[v_idx]

        t_y = y[t_idx]
        v_y = y[v_idx]

        keras.backend.clear_session()
        model = esim(embeddings=embed_matrix, gl_embed_matrix=gl_embed_matrix, maxlen=config['maxlen'], lr=0.001,
                     f_dim=test_feature.shape[1])

        model_path = config['model_path'].format(split_id)
        cb = CB(val_data=([q1_val, q2_val, q1_gl_val, q2_gl_val, f_val], v_y), model_path=model_path)
        model.fit(x=[q1_train, q2_train, q1_gl_train, q2_gl_train, f_train], y=t_y, batch_size=config['batch_size'],
                  epochs=config['epochs'],
                  validation_data=([q1_val, q2_val, q1_gl_val, q2_gl_val, f_val], v_y), verbose=True, callbacks=[cb])

        # 保存各个fold的max f1
        max_f1 = cb.get_max_f1()
        f1_record.append([split_id, max_f1])

        for item in f1_record:
            print('{}\t{}'.format(item[0], item[1]))


def train_5_round():
    # 设置内存自适应
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    KTF.set_session(session)

    f1_record = []
    for round_id in range(1, 6):
        f1_record.append(['round', round_id])
        vocabulary, embed_matrix = dh.load_embed(file_name='../data/char_embedding.txt')
        gl_vocabulary, gl_embed_matrix = dh.load_embed(file_name='../data/gl_char_vectors.txt')

        train_file_name = '../data/train.csv'
        test_file_name = '../data/test.csv'
        q1, q2, y = data_pipeline(train_file_name, vocabulary=vocabulary, return_label=True, maxlen=config['maxlen'])
        q1_gl, q2_gl = data_pipeline(train_file_name, vocabulary=gl_vocabulary, return_label=False,
                                     maxlen=config['maxlen'])

        q1_test, q2_test = data_pipeline(test_file_name, vocabulary=vocabulary, return_label=False,
                                         maxlen=config['maxlen'])
        q1_test_gl, q2_test_gl = data_pipeline(test_file_name, vocabulary=gl_vocabulary, return_label=False,
                                               maxlen=config['maxlen'])

        train_feature = pd.read_csv('../feature/train_feature.csv')
        train_feature = train_feature.values
        test_feature = pd.read_csv('../feature/test_feature.csv')
        test_feature = test_feature.values

        kf = KFold(n_splits=5)
        split_id = 0

        for t_idx, v_idx in kf.split(q1, q2, y):
            split_id += 1

            print('Round [{}] fold [{}]'.format(round_id, split_id))

            q1_train = q1[t_idx]
            q1_val = q1[v_idx]

            q2_train = q2[t_idx]
            q2_val = q2[v_idx]

            q1_gl_train = q1_gl[t_idx]
            q1_gl_val = q1_gl[v_idx]

            q2_gl_train = q2_gl[t_idx]
            q2_gl_val = q2_gl[v_idx]

            f_train = train_feature[t_idx]
            f_val = train_feature[v_idx]

            t_y = y[t_idx]
            v_y = y[v_idx]

            print('Ratio : {}'.format(np.sum(v_y) / len(v_y)))

            # 交换q1 和q2 的位置，形成新的数据
            # tmp1 = np.concatenate([q1_train, q2_train], axis=0)
            # tmp2 = np.concatenate([q2_train, q1_train], axis=0)
            # tmp3 = np.concatenate([q1_gl_train, q2_gl_train], axis=0)
            # tmp4 = np.concatenate([q2_gl_train, q1_gl_train], axis=0)
            #
            # q1_train = copy.deepcopy(tmp1)
            # q2_train = copy.deepcopy(tmp2)
            # q1_gl_train = copy.deepcopy(tmp3)
            # q2_gl_train = copy.deepcopy(tmp4)
            # f_train = np.concatenate([f_train, f_train], axis=0)
            # t_y = copy.deepcopy(np.concatenate([t_y, t_y], axis=0))

            keras.backend.clear_session()

            model = esim(embeddings=embed_matrix, gl_embed_matrix=gl_embed_matrix, maxlen=config['maxlen'], lr=0.001,
                         f_dim=test_feature.shape[1])

            model_path = '../models/esim_char_gl_feature/roung_{}_fold_{}.h5'.format(round_id, split_id)
            cb = CB(val_data=([q1_val, q2_val, q1_gl_val, q2_gl_val, f_val], v_y), model_path=model_path)
            model.fit(x=[q1_train, q2_train, q1_gl_train, q2_gl_train, f_train], y=t_y, batch_size=config['batch_size'],
                      epochs=config['epochs'],
                      validation_data=([q1_val, q2_val, q1_gl_val, q2_gl_val, f_val], v_y), verbose=True,
                      callbacks=[cb])

            # 加载最优模型
            model.load_weights(model_path)

            # 保存各个fold的max f1
            max_f1 = cb.get_max_f1()
            f1_record.append([split_id, max_f1])
            f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
            f1_record_df.to_csv('../log/5round_esim_char_gl_feature_f1.csv', index=False)
            print('Kfold [{}] max f1: {}'.format(split_id, max_f1))

            # 保存验证集上的预测结果
            v_pred = model.predict(x=[q1_val, q2_val, q1_gl_val, q2_gl_val, f_train])
            np.save('../prediction/esim_char_gl_feature/round_{}_train_kf_{}.npy'.format(round_id, split_id), v_pred)

            y_pred = model.predict(x=[q1_test, q2_test, q1_test_gl, q2_test_gl, test_feature])
            np.save('../prediction/esim_char_gl_feature/round_{}_test_kf_{}.npy'.format(round_id, split_id), y_pred)

        # 输出5round的结果
        for item in f1_record:
            print('{}\t{}'.format(item[0], item[1]))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # train()
    train_5_round()
    pass
