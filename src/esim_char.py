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


def esim(embeddings, maxlen, lstm_dim=128, spatial_dropout=0.3, dense_dropout=0.5,lr=0.001):
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    embedding = Embedding(embeddings.shape[0], embeddings.shape[-1], weights=[embeddings], input_length=maxlen,
                          trainable=False)

    q1_embed = BatchNormalization(axis=-1)(embedding(q1))
    q2_embed = BatchNormalization(axis=-1)(embedding(q2))

    # q_embed_sub = Subtract()([q1_embed, q2_embed])
    # q_embed_mul = Multiply()([q1_embed, q2_embed])
    #
    # q_embed_rep = Concatenate()([GlobalMaxPool1D()(q_embed_sub), GlobalMaxPool1D()(q_embed_mul)])

    # q1_embed = SpatialDropout1D(0.3)(q1_embed)
    # q2_embed = SpatialDropout1D(0.3)(q2_embed)

    # Encode
    encode = Bidirectional(CuDNNGRU(lstm_dim, return_sequences=True))

    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    combine_list = []
    combine_list.append(Subtract()([GlobalMaxPool1D()(q1_encoded), GlobalMaxPool1D()(q2_encoded)]))
    combine_list.append(Multiply()([GlobalMaxPool1D()(q1_encoded), GlobalMaxPool1D()(q2_encoded)]))
    combine_list.append(Subtract()([GlobalAvgPool1D()(q1_encoded), GlobalAvgPool1D()(q2_encoded)]))
    combine_list.append(Multiply()([GlobalAvgPool1D()(q1_encoded), GlobalAvgPool1D()(q2_encoded)]))
    # q_embed_sub = Subtract()([GlobalMaxPool1D()(q1_encoded), GlobalMaxPool1D()(q2_encoded)])
    # q_embed_mul = Multiply()([GlobalMaxPool1D()(q1_encoded), GlobalMaxPool1D()(q2_encoded)])

    q_embed_rep = Concatenate()(combine_list)
    q_embed_rep = BatchNormalization()(q_embed_rep)
    q_embed_rep = Dropout(0.5)(q_embed_rep)

    #
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

    q1_combined = BatchNormalization(axis=-1)(q1_combined)
    q2_combined = BatchNormalization(axis=-1)(q2_combined)

    compose = Bidirectional(CuDNNGRU(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)

    q1_compare = SpatialDropout1D(spatial_dropout)(q1_compare)
    q2_compare = SpatialDropout1D(spatial_dropout)(q2_compare)

    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    q_rep = Concatenate()([q1_rep, q2_rep])
    q_rep = BatchNormalization(axis=-1)(q_rep)
    q_rep = Dropout(0.3)(q_rep)

    # Classifier
    merged = Concatenate()([q_rep, q_embed_rep])

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

    model = Model(inputs=[q1, q2], outputs=out_)
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
    'epochs': 40,
    'log_file': '../log/esim_char_f1.csv',
    'result_file': '../submit/esim_char_v2.csv',
    'model_path': '../models/esim_char/kf_{}.h5'
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

    result = []
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
        model = esim(embeddings=embed_matrix, maxlen=config['maxlen'], lr=0.001)

        model_path = config['model_path'].format(split_id)
        cb = CB(val_data=([q1_val, q2_val], v_y), model_path=model_path)
        model.fit(x=[q1_train, q2_train], y=t_y, batch_size=config['batch_size'], epochs=config['epochs'],
                  validation_data=([q1_val, q2_val], v_y), verbose=True, callbacks=[cb])

        # 加载最优模型
        model.load_weights(model_path)

        # 保存各个fold的max f1
        max_f1 = cb.get_max_f1()
        f1_record.append([split_id, max_f1])
        f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
        f1_record_df.to_csv(config['log_file'], index=False)
        print('Kfold [{}] max f1: {}'.format(split_id, max_f1))

        # # 保存验证集上的预测结果
        # v_pred = model.predict(x=[q1_val, q2_val])
        # np.save('../prediction/esim_char/train_kf_{}.npy'.format(split_id), v_pred)
        #
        # y_pred = model.predict(x=[q1_test, q2_test])
        # np.save('../prediction/esim_char/test_kf_{}.npy'.format(split_id), y_pred)
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

        train_file_name = '../data/train.csv'
        test_file_name = '../data/test.csv'
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

            model_path = '../models/esim_char/round_{}_kf_{}.h5'.format(round_id, split_id)
            cb = CB(val_data=([q1_val, q2_val], v_y), model_path=model_path)
            model.fit(x=[q1_train, q2_train], y=t_y, batch_size=config['batch_size'], epochs=config['epochs'],
                      validation_data=([q1_val, q2_val], v_y), verbose=True, callbacks=[cb])

            # 加载最优模型
            model.load_weights(model_path)

            # 保存各个fold的max f1
            max_f1 = cb.get_max_f1()
            f1_record.append([split_id, max_f1])
            f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
            f1_record_df.to_csv('../log/5round_esim_char_f1.csv', index=False)
            print('Kfold [{}] max f1: {}'.format(split_id, max_f1))

            # 保存验证集上的预测结果
            v_pred = model.predict(x=[q1_val, q2_val])
            np.save('../prediction/esim_char/round_{}_train_kf_{}.npy'.format(round_id, split_id), v_pred)

            y_pred = model.predict(x=[q1_test, q2_test])
            np.save('../prediction/esim_char/round_{}_test_kf_{}.npy'.format(round_id, split_id), y_pred)

    # 输出5round的结果
    for item in f1_record:
        print('{}\t{}'.format(item[0], item[1]))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # train()
    train_5_round()
    pass
