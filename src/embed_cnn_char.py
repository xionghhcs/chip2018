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
    attention = Dot(axes=-1, name='attention_dot')([input_1, input_2])

    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    # 这里相当于矩阵转置了
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))

    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


class BiLinearLayer(Layer):

    def __init__(self, **kwargs):
        super(BiLinearLayer, self).__init__(**kwargs)
        pass

    def build(self, input_shape):
        if isinstance(input_shape, list):
            pass

        dim = input_shape[0][-1]

        self.U = self.add_weight(name='attention_weight', shape=[dim, dim], initializer='uniform', trainable=True)
        self.built = True
        super(BiLinearLayer, self).build(input_shape)
        pass

    def call(self, inputs, **kwargs):
        q1 = inputs[0]
        q2 = inputs[1]
        time_step = q1.shape[1].value
        dim = q1.shape[-1].value

        tmp = tf.tensordot(q1, self.U, axes=1)
        tmp = tf.reshape(tmp, shape=[-1, time_step, dim])

        att = tf.matmul(tmp, q2, transpose_b=True)

        att = tf.nn.relu(att)

        w1_att = tf.nn.softmax(att, dim=1)
        w2_att = tf.nn.softmax(att, dim=2)

        q1_align = tf.matmul(w1_att, q1, transpose_a=True)
        q2_align = tf.matmul(w2_att, q2)

        return [q1_align, q2_align]

    def compute_output_shape(self, input_shape):
        return [input_shape[1], input_shape[0]]


def esim(embeddings, maxlen, dense_dropout=0.5):
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    embedding = Embedding(embeddings.shape[0], embeddings.shape[-1], weights=[embeddings], input_length=maxlen,
                          trainable=False)

    # q1_embed = BatchNormalization(axis=-1)(embedding(q1))
    # q2_embed = BatchNormalization(axis=-1)(embedding(q2))

    q1_embed = embedding(q1)
    q2_embed = embedding(q2)

    q1_embed = SpatialDropout1D(0.2)(q1_embed)
    q2_embed = SpatialDropout1D(0.2)(q2_embed)

    # Encode
    encode = Bidirectional(CuDNNGRU(256, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # q_dot = Dot(axes=-1)([q1_encoded, q2_encoded])
    # q_dot = Reshape(target_shape=[maxlen * maxlen])(q_dot)
    # q_dot = BatchNormalization()(q_dot)
    # q_dot = Dropout(0.4)(q_dot)
    # q_dot_rep = q_dot

    q_encoded_rep = Concatenate()([Subtract()([GlobalMaxPool1D()(q1_encoded), GlobalMaxPool1D()(q2_encoded)]),
                                   Multiply()([GlobalMaxPool1D()(q1_encoded), GlobalMaxPool1D()(q2_encoded)])])
    q_encoded_rep = BatchNormalization()(q_encoded_rep)
    q_encoded_rep = Dropout(0.5)(q_encoded_rep)
    # q_dot_rep = Dense(units=256, activation='relu')(q_dot)

    # print('q_dot : ')
    # print(q_dot)

    q1_encoded = SpatialDropout1D(0.2)(q1_encoded)
    q2_encoded = SpatialDropout1D(0.2)(q2_encoded)

    # encode = Bidirectional(CuDNNGRU(128, return_sequences=True))
    # encode2 = Bidirectional(CuDNNGRU(128, return_sequences=True))
    # encode3 = Bidirectional(CuDNNGRU(128, return_sequences=True))

    # q1_encoded = Concatenate(axis=-1)([q1_encoded, q1_embed])
    # q2_encoded = Concatenate(axis=-1)([q2_encoded, q2_embed])
    #
    # q1_encoded = BatchNormalization()(q1_encoded)
    # q2_encoded = BatchNormalization()(q2_encoded)
    #
    # q1_encoded = encode2(q1_encoded)
    # q2_encoded = encode2(q2_encoded)

    # q_encoded = encode3(Concatenate()([q1_encoded, q2_encoded]))

    filter_list = [2, 3, 4, 5]

    q1_max_pool = []
    # q1_avg_pool = []
    q2_max_pool = []
    # q2_avg_pool = []

    # q_encoded = Concatenate()([q1_encoded, q2_encoded])
    filter_num = 256
    for kernel_size in filter_list:
        conv_1 = Conv1D(kernel_size=kernel_size, filters=filter_num, activation='relu')
        # conv_2 = Conv1D(kernel_size=kernel_size, filters=filter_num, activation='relu')
        q1_conv = conv_1(q1_encoded)
        q2_conv = conv_1(q2_encoded)

        q1_conv = SpatialDropout1D(0.3)(q1_conv)
        q2_conv = SpatialDropout1D(0.3)(q2_conv)

        q1_max_pool.append(GlobalMaxPool1D()(q1_conv))
        # q1_avg_pool.append(Reshape(target_shape=[1, filter_num])(GlobalAvgPool1D()(q1_conv)))

        q2_max_pool.append(GlobalMaxPool1D()(q2_conv))
        # q2_avg_pool.append(Reshape(target_shape=[1, filter_num])(GlobalAvgPool1D()(q2_conv)))

    q1_seq = Concatenate(axis=-1)(q1_max_pool)
    # q1_avg_seq = Concatenate(axis=1)(q1_avg_pool)
    # q1_seq = Concatenate(axis=1)([q1_max_seq, q1_avg_seq])

    q2_seq = Concatenate(axis=-1)(q2_max_pool)
    # q2_avg_seq = Concatenate(axis=1)(q2_avg_pool)
    # q2_seq = Concatenate(axis=1)([q2_max_seq, q2_avg_seq])

    q_sub = Subtract()([q1_seq, q2_seq])
    q_mul = Multiply()([q1_seq, q2_seq])

    q_rep = Concatenate()([q_sub, q_mul])

    q_rep = BatchNormalization()(q_rep)

    # q1_align, q2_align = soft_attention_alignment(q1_seq, q2_seq)
    #
    # q1_rep = Concatenate()([q1_seq, q2_align])
    # q2_rep = Concatenate()([q2_seq, q1_align])
    #
    # q1_rep = Concatenate()([GlobalMaxPool1D()(q1_rep), GlobalAvgPool1D()(q1_rep)])
    # q2_rep = Concatenate()([GlobalMaxPool1D()(q2_rep), GlobalAvgPool1D()(q2_rep)])

    # q1_rep = Flatten()(q_sub)
    # q2_rep = Flatten()(q_mul)

    merged = Concatenate()([q_rep, q_encoded_rep])

    print('-----------------')
    print(merged)

    dense = BatchNormalization()(merged)
    dense = Dropout(dense_dropout)(dense)
    # dense = Dense(512, activation='relu')(dense)
    # dense = BatchNormalization()(dense)
    # dense = Dropout(dense_dropout)(dense)
    dense = Dense(256, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def data_pipeline(file_name, vocabulary, maxlen=15, return_label=False):
    questions = pd.read_csv('../data/question_id.csv')
    data = pd.read_csv(file_name)

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
    'epochs': 40,
    'log_file': '../log/embed_cnn_char_f1.csv',
    'result_file': '../submit/embed_cnn_char.csv',
    'model_path': '../models/embed_cnn_char/kf_{}.h5'
}


class CB(Callback):
    def __init__(self, val_data, tolerates=5, model_path='../models/tmp.h5'):
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
        print('-val_f1:{:.4f}'.format(metrics['f1']))
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

    train_file_name = '../data/train.csv'
    test_file_name = '../data/test.csv'
    test_data = pd.read_csv(test_file_name)
    q1, q2, y = data_pipeline(train_file_name, vocabulary=vocabulary, return_label=True, maxlen=config['maxlen'])
    q1_test, q2_test = data_pipeline(test_file_name, vocabulary=vocabulary, return_label=False, maxlen=config['maxlen'])

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

        t_y = y[t_idx]
        v_y = y[v_idx]

        keras.backend.clear_session()
        model = esim(embeddings=embed_matrix, maxlen=config['maxlen'])

        model_path = config['model_path'].format(split_id)
        cb = CB(val_data=([q1_val, q2_val], v_y), model_path=model_path)
        model.fit(x=[q1_train, q2_train], y=t_y, batch_size=config['batch_size'], epochs=config['epochs'],
                  validation_data=([q1_val, q2_val], v_y), verbose=True, callbacks=[cb])

        # 加载最优模型
        model.load_weights(model_path)

        # 保存各个fold的max f1
        max_f1 = cb.get_max_f1()
        print('Fold [{}] f1: {:.4f}'.format(split_id, max_f1))
        f1_record.append([split_id, max_f1])
        f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
        f1_record_df.to_csv(config['log_file'], index=False)

        # # 保存验证集上的预测结果
        # v_pred = model.predict(x=[q1_val, q2_val])
        # np.save('../prediction/embed_cnn_char/train_kf_{}.npy'.format(split_id), v_pred)
        #
        # # 保存测试集上的预测结果
        # y_pred = model.predict(x=[q1_test, q2_test])
        # np.save('../prediction/embed_cnn_char/test_kf_{}.npy'.format(split_id), y_pred)

    for item in f1_record:
        print('{}\t{:.4f}'.format(item[0], item[1]))


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

        train_file_name = '../data/train.csv'
        test_file_name = '../data/test.csv'
        test_data = pd.read_csv(test_file_name)

        q1, q2, y = data_pipeline(train_file_name, vocabulary=vocabulary, return_label=True, maxlen=config['maxlen'])
        q1_test, q2_test = data_pipeline(test_file_name, vocabulary=vocabulary, return_label=False,
                                         maxlen=config['maxlen'])

        kf = KFold(n_splits=5)
        split_id = 0
        for t_idx, v_idx in kf.split(q1, q2, y):
            split_id += 1
            print('Round [{}] fold [{}]'.format(round_id, split_id))

            q1_train = q1[t_idx]
            q1_val = q1[v_idx]

            q2_train = q2[t_idx]
            q2_val = q2[v_idx]

            t_y = y[t_idx]
            v_y = y[v_idx]

            keras.backend.clear_session()
            model = esim(embeddings=embed_matrix, maxlen=config['maxlen'])

            model_path = '../models/embed_cnn_char/round_{}_kf{}.h5'.format(round_id, split_id)
            cb = CB(val_data=([q1_val, q2_val], v_y), model_path=model_path)
            model.fit(x=[q1_train, q2_train], y=t_y, batch_size=config['batch_size'], epochs=config['epochs'],
                      validation_data=([q1_val, q2_val], v_y), verbose=True, callbacks=[cb])

            # 加载最优模型
            model.load_weights(model_path)

            # 保存各个fold的max f1
            max_f1 = cb.get_max_f1()
            print('Round [{}] Fold [{}] f1: {:.4f}'.format(round_id, split_id, max_f1))
            f1_record.append([split_id, max_f1])
            f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
            f1_record_df.to_csv('../log/5round_embed_cnn_char_f1.csv', index=False)

            # 保存验证集上的预测结果
            v_pred = model.predict(x=[q1_val, q2_val])
            np.save('../prediction/embed_cnn_char/round_{}_train_kf_{}.npy'.format(round_id, split_id), v_pred)

            # 保存测试集上的预测结果
            y_pred = model.predict(x=[q1_test, q2_test])
            np.save('../prediction/embed_cnn_char/round_{}_test_kf_{}.npy'.format(round_id, split_id), y_pred)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # train()
    train_5_round()
    pass
