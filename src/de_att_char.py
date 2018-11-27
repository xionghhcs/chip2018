import numpy as np

from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.activations import softmax
from keras.models import *
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import Callback
from keras.models import load_model
from keras.utils import multi_gpu_model

import itertools
from time import time
import datetime
import sys
import copy
import os
import logging
import json
import data_helper as dh

from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    """
    Substract element-wise
    """
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


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
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def decomposable_attention(embeddings, maxlen, projection_dim=128, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=256, compare_dropout=0.2, dense_dropout=0.2,
                           lr=1e-3, activation='relu'):
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], input_length=maxlen,
                          trainable=False)

    q1_embed = embedding(q1)
    q2_embed = embedding(q2)

    q1_embed = SpatialDropout1D(0.3)(q1_embed)
    q2_embed = SpatialDropout1D(0.3)(q2_embed)

    lstm_encoder = Bidirectional(CuDNNGRU(units=128, return_sequences=True))

    q1_embed = lstm_encoder(q1_embed)
    q2_embed = lstm_encoder(q2_embed)

    q1_embed = SpatialDropout1D(0.3)(q1_embed)
    q2_embed = SpatialDropout1D(0.3)(q2_embed)

    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=None),
            Dropout(rate=projection_dropout),
        ])

    projection_layers.extend([
        Dense(projection_dim, activation=None),
        # Dropout(rate=projection_dropout),
    ])

    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    q1_combined = BatchNormalization(axis=-1)(q1_combined)
    q2_combined = BatchNormalization(axis=-1)(q2_combined)
    # compare_layers = [
    #     Dense(compare_dim, activation=activation),
    #     Dropout(compare_dropout),
    #     Dense(compare_dim, activation=activation),
    #     Dropout(compare_dropout),
    # ]

    encoder2 = Bidirectional(CuDNNGRU(128, return_sequences=True))

    q1_compare = encoder2(q1_combined)
    q2_compare = encoder2(q2_combined)

    # q1_compare = time_distributed(q1_combined, compare_layers)
    # q2_compare = time_distributed(q2_combined, compare_layers)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dropout(0.5)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
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
    'batch_size': 128,
    'epochs': 40,
    'log_path': '../log/de_att_char_f1.csv',
    'model_path': '../models/de_att_char/kf{}.h5',
    'result_path': '../submit/de_att_char_v1.csv'
}


class CB(Callback):
    def __init__(self, val_data, tolerates=4, model_path='../models/tmp.h5'):
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

    print(q1.shape)
    print(y.shape)
    print(y[:20])

    result = []
    kf = KFold(n_splits=5)
    split_id = 0
    f1_record = []
    for t_idx, v_idx in kf.split(q1, q2, y):
        split_id += 1
        print('## KFOLD [{}]'.format(split_id))

        q1_train = q1[t_idx]
        q1_val = q1[v_idx]

        q2_train = q2[t_idx]
        q2_val = q2[v_idx]

        train_y = y[t_idx]
        val_y = y[v_idx]

        model = decomposable_attention(embeddings=embed_matrix, maxlen=config['maxlen'])
        model_path = config['model_path'].format(split_id)
        cb = CB(val_data=([q1_val, q2_val], val_y), model_path=model_path)

        model.fit(x=[q1_train, q2_train], y=train_y, batch_size=config['batch_size'], epochs=config['epochs'],
                  validation_data=([q1_val, q2_val], val_y), verbose=True, callbacks=[cb])
        model.load_weights(model_path)

        max_f1 = cb.get_max_f1()
        print('kfold [{}] max f1: {}'.format(split_id, max_f1))
        f1_record.append([split_id, max_f1])
        f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
        f1_record_df.to_csv(config['log_path'], index=False)

        # # 验证集上的预测结果
        # v_pred = model.predict([q1_val, q2_val])
        # np.save('../prediction/de_att_char/train_kf_{}.npy'.format(split_id), v_pred)
        #
        # y_pred = model.predict([q1_test, q2_test])
        # np.save('../prediction/de_att_char/test_kf_{}.npy'.format(split_id), y_pred)


def train_5_round():
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

            train_y = y[t_idx]
            val_y = y[v_idx]

            model = decomposable_attention(embeddings=embed_matrix, maxlen=config['maxlen'])
            model_path = '../models/de_att_char/round_{}_kf_{}'.format(round_id, split_id)
            cb = CB(val_data=([q1_val, q2_val], val_y), model_path=model_path)

            model.fit(x=[q1_train, q2_train], y=train_y, batch_size=config['batch_size'], epochs=config['epochs'],
                      validation_data=([q1_val, q2_val], val_y), verbose=True, callbacks=[cb])
            model.load_weights(model_path)

            max_f1 = cb.get_max_f1()
            print('Round [{}] fold [{}] max f1: {}'.format(round_id, split_id, max_f1))
            f1_record.append([split_id, max_f1])
            f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
            f1_record_df.to_csv('../log/5round_de_att_char_f1.csv', index=False)

            # 验证集上的预测结果
            v_pred = model.predict([q1_val, q2_val])
            np.save('../prediction/de_att_char/round_{}_train_kf_{}.npy'.format(round_id, split_id), v_pred)

            y_pred = model.predict([q1_test, q2_test])
            np.save('../prediction/de_att_char/round_{}_test_kf_{}.npy'.format(round_id, split_id), y_pred)
        pass


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()
    # train_5_round()
