import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from sklearn.metrics import f1_score, accuracy_score


def get_model(maxlen, embed_matrix, lr=0.001):
    q1_input = Input(shape=(maxlen,))
    q2_input = Input(shape=(maxlen,))

    # encoder

    text_input = Input(shape=(maxlen,))

    embedding = Embedding(embed_matrix.shape[0], embed_matrix.shape[1], weights=[embed_matrix], input_length=maxlen,
                          trainable=False)

    x = embedding(text_input)
    # x = TimeDistributed(Dense(units=150, activation='relu'))(x)

    xlstm = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    xlstm1 = GlobalMaxPool1D()(xlstm)

    xa = Concatenate()([xlstm, x])
    xa_dropout = Dropout(0.3)(xa)

    xgru = Bidirectional(CuDNNGRU(256, return_sequences=True))(xa_dropout)

    conv_result = []
    for filter_size in [1, 2, 3, 4, 5]:
        if filter_size in [3, 4, 5]:
            conv = Convolution1D(filters=100, kernel_size=filter_size, padding='same', activation='relu',
                                 dilation_rate=2)(xa)
        else:
            conv = Convolution1D(filters=100, kernel_size=filter_size, padding='same', activation='relu')(xa)
        conv = GlobalMaxPool1D()(conv)
        conv_result.append(conv)
    conv_result.append(xlstm1)

    conv_result = Concatenate()(conv_result)
    # conv_result = BatchNormalization()(conv_result)
    conv_result = Dropout(0.5)(conv_result)
    conv_result = Dense(units=256, activation='relu')(conv_result)
    # conv_result = PReLU()(conv_result)

    text_encoder = Model(inputs=text_input, outputs=[conv_result, xlstm, xgru])

    x1, l, lc = text_encoder(q1_input)
    x2, r, rc = text_encoder(q2_input)

    rep1 = Dot(axes=[2, 2], normalize=True)([l, r])
    rep1 = Reshape((-1,))(rep1)
    rep1 = Dropout(0.5)(rep1)
    rep1 = Dense(units=200, activation='relu')(rep1)
    # rep1 = PReLU()(rep1)

    rep2 = Dot(axes=[2, 2], normalize=True)([lc, rc])
    rep2 = Reshape((-1,))(rep2)
    rep2 = Dropout(0.5)(rep2)
    rep2 = Dense(units=200, activation='relu')(rep2)
    # rep2 = PReLU()(rep2)

    rep3 = Concatenate()([Subtract()([x1, x2]), Multiply()([x1, x2]), x1, x2])
    # rep3 = Dropout(0.5)(rep3)
    # rep3 = Dense(units=200)(rep3)
    # rep3 = PReLU()(rep3)

    merged = Concatenate()([rep1, rep2, rep3])
    print('-------')
    print(merged)

    merged = BatchNormalization()(merged)
    dense = Dense(units=256, activation='relu')(merged)
    # dense = PReLU()(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)

    dense = Dense(64, activation='relu')(dense)
    # dense = PReLU()(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    out_ = Dense(units=1, activation='sigmoid')(dense)

    model = Model(inputs=[q1_input, q2_input], outputs=out_)

    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

    return model


import pandas as pd
import numpy as np
import keras.backend.tensorflow_backend as KTF
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
import os
import sys
import time
import datetime
import data_helper as dh


def data_pipeline(file_name, vocabulary, maxlen=15, return_label=False):
    questions = pd.read_csv('../data/question_id.csv')
    data = pd.read_csv(file_name)

    tmp = pd.merge(data, questions, left_on='qid1', right_on='qid', how='left')
    q1 = list(tmp['wid'].values)

    tmp = pd.merge(data, questions, left_on='qid2', right_on='qid', how='left')
    q2 = list(tmp['wid'].values)

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


config = {
    'maxlen': 15,
    'batch_size': 256,
    'epochs': 40,
    'model_path': '../models/bimpm_char/kf_{}.h5',
    'result_file': '../submit/bimpm_char.csv',
    'log_file': '../log/bimpm_char_f1.csv'
}

from sklearn.model_selection import KFold


def train():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    KTF.set_session(session)

    vocabulary, embed_matrix = dh.load_embed(file_name='../data/char_embedding.txt')

    train_file_name = '../data/train.csv'
    test_file_name = '../data/test.csv'

    q1, q2, y = data_pipeline(train_file_name, vocabulary=vocabulary, return_label=True, maxlen=config['maxlen'])

    q1_test, q2_test = data_pipeline(test_file_name, vocabulary=vocabulary, return_label=False, maxlen=config['maxlen'])

    kf = KFold(n_splits=5)
    f1_record = []
    split_id = 0
    for t_idx, v_idx in kf.split(q1, q2, y):
        split_id += 1

        q1_train = q1[t_idx]
        q1_val = q1[v_idx]

        q2_train = q2[t_idx]
        q2_val = q2[v_idx]

        train_y = y[t_idx]
        val_y = y[v_idx]

        # 设置内存自适应
        keras.backend.clear_session()

        model = get_model(embed_matrix=embed_matrix, maxlen=config['maxlen'])

        model_path = '../models/tmp.h5'.format(split_id)
        cb = CB(val_data=([q1_val, q2_val], val_y), model_path=model_path)
        model.fit(x=[q1_train, q2_train], y=train_y, batch_size=config['batch_size'], epochs=config['epochs'],
                  validation_data=([q1_val, q2_val], val_y), callbacks=[cb])

        # 保存f1
        max_f1 = cb.get_max_f1()
        f1_record.append([split_id, max_f1])
        f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
        # f1_record_df.to_csv(config['log_file'], index=False)

        # # 验证集上的预测结果
        # model.load_weights(model_path)
        # v_pred = model.predict([q1_val, q2_val])
        # np.save('../prediction/bimpm_char/train_kf_{}.npy'.format(split_id), v_pred)
        #
        # # 测试集合上预测结果
        # y_pred = model.predict([q1_test, q2_test])
        # np.save('../prediction/bimpm_char/test_kf_{}.npy'.format(split_id), y_pred)
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

        vocabulary, embed_matrix = dh.load_embed('../data/word_embedding.txt')

        train_file_name = '../data/train.csv'
        test_file_name = '../data/test.csv'
        q1, q2, y = data_pipeline(train_file_name, vocabulary=vocabulary, return_label=True, maxlen=config['maxlen'])
        q1_test, q2_test = data_pipeline(test_file_name, vocabulary=vocabulary, return_label=False,
                                         maxlen=config['maxlen'])

        kf = KFold(n_splits=5)
        split_id = 0
        for t_idx, v_idx in kf.split(q1, q2, y):
            split_id += 1

            print('Round [{}] Kfold [{}]'.format(round_id, split_id))

            q1_train = q1[t_idx]
            q1_val = q1[v_idx]

            q2_train = q2[t_idx]
            q2_val = q2[v_idx]

            t_y = y[t_idx]
            v_y = y[v_idx]

            keras.backend.clear_session()

            model_path = '../models/my_nn_word/round_{}_kf_{}.h5'.format(round_id, split_id)
            cb = CB(val_data=([q1_val, q2_val], v_y), model_path=model_path)

            model = get_model(maxlen=config['maxlen'], embed_matrix=embed_matrix)

            model.fit(x=[q1_train, q2_train], y=t_y, batch_size=config['batch_size'], epochs=config['epochs'],
                      validation_data=([q1_val, q2_val], v_y), verbose=True, callbacks=[cb])

            max_f1 = cb.get_max_f1()
            f1_record.append([split_id, max_f1])
            f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
            f1_record_df.to_csv('../log/5round_my_nn_word_f1.csv', index=False)
            print('fold [{}] f1 [{}]'.format(split_id, max_f1))

            # 保存验证集上的预测结果
            v_pred = model.predict(x=[q1_val, q2_val])
            np.save('../prediction/my_nn_word/round_{}_train_kf_{}.npy'.format(round_id, split_id), v_pred)

            # 保存测试集上的预测结果
            y_pred = model.predict(x=[q1_test, q2_test])
            np.save('../prediction/my_nn_word/round_{}_test_kf_{}.npy'.format(round_id, split_id), y_pred)

            for item in f1_record:
                print('{}\t{}'.format(item[0], item[1]))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # train()
    train_5_round()