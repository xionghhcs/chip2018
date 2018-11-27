import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.layers.merge import concatenate
import keras.backend as K
from keras.layers import *
from keras.engine.topology import Layer
from keras.optimizers import Adam
from layers import (
    WordRepresLayer, CharRepresLayer, ContextLayer, PredictLayer
)
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.metrics import f1_score, accuracy_score
import data_helper as dh
import os
import sys
import time
import datetime


class MultiPerspective(Layer):
    """Multi-perspective Matching Layer.

    # Arguments
        mp_dim: single forward/backward multi-perspective dimention
    """

    def __init__(self, mp_dim, epsilon=1e-6, **kwargs):
        self.mp_dim = mp_dim
        self.epsilon = 1e-6
        self.strategy = 4
        super(MultiPerspective, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        embedding_size = input_shape[-1] // 2
        print('#############')
        print(embedding_size)
        # Create a trainable weight variable for this layer.
        # input_shape is bidirectional RNN input shape
        # kernel shape (mp_dim * 2 * self.strategy, embedding_size)
        self.kernel = self.add_weight((self.mp_dim,
                                       embedding_size * 2 * self.strategy),
                                      name='kernel',
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernel_full_fw = self.kernel[:, :embedding_size]
        self.kernel_full_bw = self.kernel[:, embedding_size: embedding_size * 2]
        self.kernel_attentive_fw = self.kernel[:, embedding_size * 2: embedding_size * 3]
        self.kernel_attentive_bw = self.kernel[:, embedding_size * 3: embedding_size * 4]
        self.kernel_max_attentive_fw = self.kernel[:, embedding_size * 4: embedding_size * 5]
        self.kernel_max_attentive_bw = self.kernel[:, embedding_size * 5: embedding_size * 6]
        self.kernel_max_pool_fw = self.kernel[:, embedding_size * 6: embedding_size * 7]
        self.kernel_max_pool_bw = self.kernel[:, embedding_size * 7:]
        self.built = True
        super(MultiPerspective, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], self.mp_dim * 2 * self.strategy)

    def get_config(self):
        config = {'mp_dim': self.mp_dim,
                  'epsilon': self.epsilon}
        base_config = super(MultiPerspective, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        # h1, h2: bidirectional LSTM hidden states, include forward and backward states
        #         (batch_size, timesteps, embedding_size * 2)
        h1 = inputs[0]
        h2 = inputs[1]
        embedding_size = K.int_shape(h1)[-1] // 2
        h1_fw = h1[:, :, :embedding_size]
        h1_bw = h1[:, :, embedding_size:]
        h2_fw = h2[:, :, :embedding_size]
        h2_bw = h2[:, :, embedding_size:]

        # 4 matching strategy
        list_matching = []

        # full matching ops
        matching_fw = self._full_matching(h1_fw, h2_fw, self.kernel_full_fw)
        matching_bw = self._full_matching(h1_bw, h2_bw, self.kernel_full_bw)
        list_matching.extend([matching_fw, matching_bw])

        # cosine matrix
        cosine_matrix_fw = self._cosine_matrix(h1_fw, h2_fw)
        cosine_matrix_bw = self._cosine_matrix(h1_bw, h2_bw)

        # attentive matching ops
        matching_fw = self._attentive_matching(
            h1_fw, h2_fw, cosine_matrix_fw, self.kernel_attentive_fw)
        matching_bw = self._attentive_matching(
            h1_bw, h2_bw, cosine_matrix_bw, self.kernel_attentive_bw)
        list_matching.extend([matching_fw, matching_bw])

        # max attentive matching ops
        matching_fw = self._max_attentive_matching(
            h1_fw, h2_fw, cosine_matrix_fw, self.kernel_max_attentive_fw)
        matching_bw = self._max_attentive_matching(
            h1_bw, h2_bw, cosine_matrix_bw, self.kernel_max_attentive_bw)
        list_matching.extend([matching_fw, matching_bw])

        # max pooling matching ops
        matching_fw = self._max_pooling_matching(h1_fw, h2_fw, self.kernel_max_pool_fw)
        matching_bw = self._max_pooling_matching(h1_bw, h2_bw, self.kernel_max_pool_bw)
        list_matching.extend([matching_fw, matching_bw])

        return K.concatenate(list_matching, axis=-1)

    def _cosine_similarity(self, x1, x2):
        """Compute cosine similarity.

        # Arguments:
            x1: (..., embedding_size)
            x2: (..., embedding_size)
        """
        cos = K.sum(x1 * x2, axis=-1)
        x1_norm = K.sqrt(K.maximum(K.sum(K.square(x1), axis=-1), self.epsilon))
        x2_norm = K.sqrt(K.maximum(K.sum(K.square(x2), axis=-1), self.epsilon))
        cos = cos / x1_norm / x2_norm
        return cos

    def _cosine_matrix(self, x1, x2):
        """Cosine similarity matrix.

        Calculate the cosine similarities between each forward (or backward)
        contextual embedding h_i_p and every forward (or backward)
        contextual embeddings of the other sentence

        # Arguments
            x1: (batch_size, x1_timesteps, embedding_size)
            x2: (batch_size, x2_timesteps, embedding_size)

        # Output shape
            (batch_size, x1_timesteps, x2_timesteps)
        """
        # expand h1 shape to (batch_size, x1_timesteps, 1, embedding_size)
        x1 = K.expand_dims(x1, axis=2)
        # expand x2 shape to (batch_size, 1, x2_timesteps, embedding_size)
        x2 = K.expand_dims(x2, axis=1)
        # cosine matrix (batch_size, h1_timesteps, h2_timesteps)
        cos_matrix = self._cosine_similarity(x1, x2)
        return cos_matrix

    def _mean_attentive_vectors(self, x2, cosine_matrix):
        """Mean attentive vectors.

        Calculate mean attentive vector for the entire sentence by weighted
        summing all the contextual embeddings of the entire sentence

        # Arguments
            x2: sequence vectors, (batch_size, x2_timesteps, embedding_size)
            cosine_matrix: cosine similarities matrix of x1 and x2,
                           (batch_size, x1_timesteps, x2_timesteps)

        # Output shape
            (batch_size, x1_timesteps, embedding_size)
        """
        # (batch_size, x1_timesteps, x2_timesteps, 1)
        expanded_cosine_matrix = K.expand_dims(cosine_matrix, axis=-1)
        # (batch_size, 1, x2_timesteps, embedding_size)
        x2 = K.expand_dims(x2, axis=1)
        # (batch_size, x1_timesteps, embedding_size)
        weighted_sum = K.sum(expanded_cosine_matrix * x2, axis=2)
        # (batch_size, x1_timesteps, 1)
        sum_cosine = K.expand_dims(K.sum(cosine_matrix, axis=-1) + self.epsilon, axis=-1)
        # (batch_size, x1_timesteps, embedding_size)
        attentive_vector = weighted_sum / sum_cosine
        return attentive_vector

    def _max_attentive_vectors(self, x2, cosine_matrix):
        """Max attentive vectors.

        Calculate max attentive vector for the entire sentence by picking
        the contextual embedding with the highest cosine similarity
        as the attentive vector.

        # Arguments
            x2: sequence vectors, (batch_size, x2_timesteps, embedding_size)
            cosine_matrix: cosine similarities matrix of x1 and x2,
                           (batch_size, x1_timesteps, x2_timesteps)

        # Output shape
            (batch_size, x1_timesteps, embedding_size)
        """
        # (batch_size, x1_timesteps)
        max_x2_step = K.argmax(cosine_matrix, axis=-1)

        embedding_size = K.int_shape(x2)[-1]
        timesteps = K.int_shape(max_x2_step)[-1]
        if timesteps is None:
            timesteps = K.shape(max_x2_step)[-1]

        # collapse time dimension and batch dimension together
        # collapse x2 to (batch_size * x2_timestep, embedding_size)
        x2 = K.reshape(x2, (-1, embedding_size))
        # collapse max_x2_step to (batch_size * h1_timesteps)
        max_x2_step = K.reshape(max_x2_step, (-1,))
        # (batch_size * x1_timesteps, embedding_size)
        max_x2 = K.gather(x2, max_x2_step)
        # reshape max_x2, (batch_size, x1_timesteps, embedding_size)
        attentive_vector = K.reshape(max_x2, K.stack([-1, timesteps, embedding_size]))
        return attentive_vector

    def _time_distributed_multiply(self, x, w):
        """Element-wise multiply vector and weights.

        # Arguments
            x: sequence of hidden states, (batch_size, ?, embedding_size)
            w: weights of one matching strategy of one direction,
               (mp_dim, embedding_size)

        # Output shape
            (?, mp_dim, embedding_size)
        """
        # dimension of vector
        n_dim = K.ndim(x)
        embedding_size = K.int_shape(x)[-1]
        timesteps = K.int_shape(x)[1]
        if timesteps is None:
            timesteps = K.shape(x)[1]

        # collapse time dimension and batch dimension together
        x = K.reshape(x, (-1, embedding_size))
        # reshape to (?, 1, embedding_size)
        x = K.expand_dims(x, axis=1)
        # reshape weights to (1, mp_dim, embedding_size)
        w = K.expand_dims(w, axis=0)
        # element-wise multiply
        x = x * w
        # reshape to original shape
        if n_dim == 3:
            x = K.reshape(x, K.stack([-1, timesteps, self.mp_dim, embedding_size]))
            x.set_shape([None, None, None, embedding_size])
        elif n_dim == 2:
            x = K.reshape(x, K.stack([-1, self.mp_dim, embedding_size]))
            x.set_shape([None, None, embedding_size])
        return x

    def _full_matching(self, h1, h2, w):
        """Full matching operation.

        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            w: weights of one direction, (mp_dim, embedding_size)

        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h2 forward last step hidden vector, (batch_size, embedding_size)
        h2_last_state = h2[:, -1, :]
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # h2_last_state * weights, (batch_size, mp_dim, embedding_size)
        h2 = self._time_distributed_multiply(h2_last_state, w)
        # reshape to (batch_size, 1, mp_dim, embedding_size)
        h2 = K.expand_dims(h2, axis=1)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, h2)
        return matching

    def _max_pooling_matching(self, h1, h2, w):
        """Max pooling matching operation.

        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            w: weights of one direction, (mp_dim, embedding_size)

        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # h2 * weights, (batch_size, h2_timesteps, mp_dim, embedding_size)
        h2 = self._time_distributed_multiply(h2, w)
        # reshape v1 to (batch_size, h1_timesteps, 1, mp_dim, embedding_size)
        h1 = K.expand_dims(h1, axis=2)
        # reshape v1 to (batch_size, 1, h2_timesteps, mp_dim, embedding_size)
        h2 = K.expand_dims(h2, axis=1)
        # cosine similarity, (batch_size, h1_timesteps, h2_timesteps, mp_dim)
        cos = self._cosine_similarity(h1, h2)
        # (batch_size, h1_timesteps, mp_dim)
        matching = K.max(cos, axis=2)
        return matching

    def _attentive_matching(self, h1, h2, cosine_matrix, w):
        """Attentive matching operation.

        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            cosine_matrix: weights of hidden state h2,
                          (batch_size, h1_timesteps, h2_timesteps)
            w: weights of one direction, (mp_dim, embedding_size)

        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # attentive vector (batch_size, h1_timesteps, embedding_szie)
        attentive_vec = self._mean_attentive_vectors(h2, cosine_matrix)
        # attentive_vec * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        attentive_vec = self._time_distributed_multiply(attentive_vec, w)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, attentive_vec)
        return matching

    def _max_attentive_matching(self, h1, h2, cosine_matrix, w):
        """Max attentive matching operation.

        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            cosine_matrix: weights of hidden state h2,
                          (batch_size, h1_timesteps, h2_timesteps)
            w: weights of one direction, (mp_dim, embedding_size)

        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # max attentive vector (batch_size, h1_timesteps, embedding_szie)
        max_attentive_vec = self._max_attentive_vectors(h2, cosine_matrix)
        # max_attentive_vec * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        max_attentive_vec = self._time_distributed_multiply(max_attentive_vec, w)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, max_attentive_vec)
        return matching


def BiMPM(embedding_matrix=None,
          sequence_length=None,
          mp_dim=20):
    # Model words input
    w1 = Input(shape=(sequence_length,), dtype='int32')
    w2 = Input(shape=(sequence_length,), dtype='int32')

    embed_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                            input_length=sequence_length, trainable=False)

    w_res1 = embed_layer(w1)
    w_res2 = embed_layer(w2)

    # BatchNormalization
    w_res1 = BatchNormalization(axis=-1)(w_res1)
    w_res2 = BatchNormalization(axis=-1)(w_res2)

    # w_res1 = Dropout(0.3)(w_res1)
    # w_res2 = Dropout(0.3)(w_res2)

    sequence1 = w_res1
    sequence2 = w_res2

    encoder_layer = Bidirectional(CuDNNGRU(units=128, return_sequences=True))
    encoder_layer2 = Bidirectional(CuDNNLSTM(units=64, return_sequences=True))
    context1 = encoder_layer(sequence1)
    context2 = encoder_layer(sequence2)

    context1_ = encoder_layer2(sequence1)
    context2_ = encoder_layer2(sequence2)

    # context1 = Dropout(0.1)(context1)
    # context2 = Dropout(0.1)(context2)
    # context1_ = Dropout(0.1)(context1_)
    # context2_ = Dropout(0.1)(context2_)

    context1 = Concatenate()([context1, context1_])
    context2 = Concatenate()([context2, context2_])

    context1 = BatchNormalization()(context1)
    context2 = BatchNormalization()(context2)

    # context1 = SpatialDropout1D(0.3)(context1)
    # context2 = SpatialDropout1D(0.3)(context2)

    context1 = Dropout(0.3)(context1)
    context2 = Dropout(0.3)(context2)

    # context1 = Concatenate(axis=-1)([w_res1, context1])
    # context2 = Concatenate(axis=-1)([w_res2, context2])
    #
    # context1 = encoder_layer2(context1)
    # context2 = encoder_layer2(context2)

    # context1_rep = GlobalMaxPool1D()(context1)
    # context2_rep = GlobalMaxPool1D()(context2)
    #
    # context_rep = Concatenate()([context1_rep, context2_rep, Subtract()([context1_rep, context2_rep]),
    #                              Multiply()([context1_rep, context2_rep])])
    #
    # context_rep = BatchNormalization()(context_rep)
    # context_rep = Dense(units=256, activation='relu')(context_rep)
    # context_rep = Dropout(0.5)(context_rep)

    # print(context1)

    # context1 = context_layer(sequence1)
    # context2 = context_layer(sequence2)

    # context1 = Dropout(0.3)(context1)
    # context2 = Dropout(0.3)(context2)

    # Build matching layer
    matching_layer = MultiPerspective(mp_dim)

    matching1 = matching_layer([context1, context2])
    matching2 = matching_layer([context2, context1])
    matching = concatenate([matching1, matching2])

    print('#############')
    print(matching)
    print('##############')

    matching = BatchNormalization(axis=-1)(matching)
    # matching = Dropout(0.3)(matching)
    # matching_rep = GlobalMaxPool1D()(matching)
    # matching_rep = Dropout(0.5)(matching_rep)
    # Build aggregation layer
    aggregate_layer = Bidirectional(CuDNNGRU(units=256, return_sequences=True))
    aggregation = aggregate_layer(matching)

    agg_avg = GlobalAveragePooling1D()(aggregation)
    agg_max = GlobalMaxPool1D()(aggregation)

    aggregation = Concatenate(axis=-1)([agg_avg, agg_max])
    print('------------')
    print(aggregation)
    dense = BatchNormalization(axis=-1)(aggregation)
    dense = Dropout(0.5)(dense)
    dense = Dense(units=128, activation='relu', kernel_regularizer=regularizers.l1(0.001))(dense)
    dense = Dropout(0.5)(dense)
    _out = Dense(units=1, activation='sigmoid')(dense)

    model = Model(inputs=[w1, w2],
                  outputs=_out)

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model


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
    def __init__(self, val_data, tolerates=7, model_path='../models/tmp.h5'):
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
    'batch_size': 128,
    'epochs': 30,
    'model_path': '../models/bimpm_word/kf_{}.h5',
    'result_file': '../submit/bimpm_word.csv',
    'log_file': '../log/bimpm_word_f1.csv',
}

from sklearn.model_selection import KFold


def train():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    KTF.set_session(session)

    vocabulary, embed_matrix = dh.load_embed(file_name='../data/word_embedding.txt')

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

        q1_train = q1[t_idx]
        q1_val = q1[v_idx]

        q2_train = q2[t_idx]
        q2_val = q2[v_idx]

        train_y = y[t_idx]
        val_y = y[v_idx]

        keras.backend.clear_session()

        model = BiMPM(embedding_matrix=embed_matrix, sequence_length=config['maxlen'])

        model_path = config['model_path'].format(split_id)
        cb = CB(val_data=([q1_val, q2_val], val_y), model_path=model_path)
        model.fit(x=[q1_train, q2_train], y=train_y, batch_size=config['batch_size'], epochs=config['epochs'],
                  validation_data=([q1_val, q2_val], val_y), callbacks=[cb])

        max_f1 = cb.get_max_f1()

        print('KF [{}] f1 : {}'.format(split_id, max_f1))

        f1_record.append([split_id, max_f1])
        f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
        f1_record_df.to_csv(config['log_file'], index=False)

        # # 保存k折上的预测结果
        # model.load_weights(model_path)
        # v_pred = model.predict([q1_val, q2_val])
        # np.save('../prediction/bimpm_word/train_kf_{}.npy'.format(split_id), v_pred)
        #
        # # 保存测试集上的预测结果
        # y_pred = model.predict([q1_test, q2_test])
        # np.save('../prediction/bimpm_word/test_kf_{}.npy'.format(split_id), y_pred)
    for item in f1_record:
        print('{}\t{}'.format(item[0], item[1]))


def train_5_round():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    KTF.set_session(session)

    f1_record = []
    for round_id in range(1, 6):
        f1_record.append(['round', round_id])
        vocabulary, embed_matrix = dh.load_embed(file_name='../data/word_embedding.txt')

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

            train_y = y[t_idx]
            val_y = y[v_idx]

            keras.backend.clear_session()
            model = BiMPM(embedding_matrix=embed_matrix, sequence_length=config['maxlen'])

            model_path = '../models/bimpm_word/round_{}_kf_{}.h5'.format(round_id, split_id)
            cb = CB(val_data=([q1_val, q2_val], val_y), model_path=model_path)
            model.fit(x=[q1_train, q2_train], y=train_y, batch_size=config['batch_size'], epochs=config['epochs'],
                      validation_data=([q1_val, q2_val], val_y), callbacks=[cb])

            max_f1 = cb.get_max_f1()

            print('KF [{}] f1 : {}'.format(split_id, max_f1))

            f1_record.append([split_id, max_f1])
            f1_record_df = pd.DataFrame(f1_record, columns=['fold', 'f1'])
            f1_record_df.to_csv('../log/5round_bimpm_word_f1.csv', index=False)

            # 保存k折上的预测结果
            model.load_weights(model_path)
            v_pred = model.predict([q1_val, q2_val])
            np.save('../prediction/bimpm_word/round_{}_train_kf_{}.npy'.format(round_id, split_id), v_pred)

            # 保存测试集上的预测结果
            y_pred = model.predict([q1_test, q2_test])
            np.save('../prediction/bimpm_word/round_{}_test_kf_{}.npy'.format(round_id, split_id), y_pred)
        pass
    for item in f1_record:
        print('{}\t{}'.format(item[0], item[1]))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train()
    # train_5_round()
    pass
