from nltk import ngrams
from simhash import Simhash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler
from gensim.models import KeyedVectors
from fuzzywuzzy import fuzz

import gensim
from gensim import corpora, models, similarities
from gensim.models import KeyedVectors
from collections import Counter
import sys
import time
import datetime
import networkx as nx

import warnings
import copy

warnings.filterwarnings('ignore')

minkowski_dis = DistanceMetric.get_metric('minkowski')


class FeatureExtractor:
    def __init__(self, train_df, test_df, w_embed_file, c_embed_file):
        self.train_df = train_df
        self.test_df = test_df

        corpus = []
        corpus += list(train_df['q1_w'].values) + list(train_df['q2_w'].values) + list(test_df['q1_w']) + list(
            test_df['q2_w'])
        # print(corpus[:2])

        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(corpus)

        # build dictionary
        sentences = []
        for s in corpus:
            sentences.append(s.split())

        self.dictionary = corpora.Dictionary(sentences)

        self.w_word2vec_model = KeyedVectors.load_word2vec_format(w_embed_file)
        self.c_word2vec_model = KeyedVectors.load_word2vec_format(c_embed_file)

        self.bm25_model, self.bm25_dim, self.bm25_avg_idf = self._init_bm25()

    def _init_bm25(self):
        questions = pd.read_csv('../data/question_id.csv')
        corpus = questions.wid.tolist()
        bm25_dim = len(corpus)
        corpus = [s.split() for s in corpus]
        from gensim.summarization.bm25 import get_bm25_weights, BM25
        bm25_model = BM25(corpus)

        bm25_avg_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())

        return bm25_model, bm25_dim, bm25_avg_idf

    def normalization(self, train_df, test_df):
        all_df = pd.concat([train_df, test_df], axis=0)
        columns = list(train_df.columns)

        for col in ['qid1', 'qid2', 'label', 'q1_w', 'q1_c', 'q2_w', 'q2_c']:
            if col in columns:
                columns.remove(col)

        for col in columns:
            try:
                scaler = MinMaxScaler()
                all_df[col] = scaler.fit_transform(all_df[col].values.reshape(-1, 1))
            except:
                print('{} error'.format(col))
                sys.exit()

        train_df, test_df = all_df.iloc[:len(train_df)], all_df.iloc[len(train_df):]
        # print(train_df.head())
        # print('train_df size:')
        # print(len(train_df))
        # print('test_df size:')
        # print(len(test_df))
        return train_df, test_df

    def extract_all_feature(self, norm=True):
        s_time = time.time()
        for df in [self.train_df, self.test_df]:
            # print('Extract hash feature...')
            # self.extract_hash_feature(df)
            # print('Extract length feature...')
            # self.extract_length_feature(df)
            # print('Extract distance feature...')
            # self.extract_distance_feature(df)
            # print('Extract word2vec feature...')
            # self.extract_word2vec_feature(df, 'w')
            # self.extract_word2vec_feature(df, 'c')
            # print('Extract tfidf feature...')
            # self.extract_tfidf_feature(df)
            # print('Extract intersection feature...')
            # self.extract_intersection_feature(df)
            # print('Extract share feature...')
            # self.extract_share_feature(df)
            # print('Extract fuzzywuzzy feature...')
            # self.extract_fuzzywuzzy_feature(df)
            # print('Extract graph feature...')
            # self.extract_graph_feature(df)
            # print('Extract edit dist feature...')
            # self.extract_edit_dist_feature(df)
            print('Extract word pair feature...')
            self.extract_word_pair_feature(df)
            # print('Extract bm25 feature...')
            # self.extract_bm25_feature(df)

        print('Extract magic feature')
        self.extract_magic_feautre()

        if norm is True:
            print('Normalization...')
            self.train_df, self.test_df = self.normalization(self.train_df, self.test_df)
        # remove columns
        self.train_df = self.train_df.drop(['qid1', 'qid2', 'label', 'q1_w', 'q1_c', 'q2_w', 'q2_c'], axis=1)
        self.test_df = self.test_df.drop(['qid1', 'qid2', 'label', 'q1_w', 'q1_c', 'q2_w', 'q2_c'], axis=1)
        e_time = time.time()

        # report the feature information:
        print('Features :')
        print(list(self.train_df.columns))

        print('Feature dim:')
        print(len(self.train_df.columns))

        print('Feature extract cost time:')
        print(datetime.timedelta(seconds=e_time - s_time))

        return self.train_df, self.test_df

    def extract_hash_feature(self, df):
        def get_ngrams(sequence, n=2):
            return ['_'.join(ngram) for ngram in ngrams(sequence, n)]

        def calculate_simhash_dist(sequence1, sequence2):
            return Simhash(sequence1).distance(Simhash(sequence2))

        def calculate_all_simhash(row):
            q1_w, q1_c = row['q1_w'].split(), row['q1_c'].split()
            q2_w, q2_c = row['q2_w'].split(), row['q2_c'].split()

            simhash_w1gram_dist = calculate_simhash_dist(q1_w, q2_w)
            simhash_w2gram_dist = calculate_simhash_dist(get_ngrams(q1_w, 2), get_ngrams(q2_w, 2))
            simhash_w3gram_dist = calculate_simhash_dist(get_ngrams(q1_w, 3), get_ngrams(q2_w, 3))

            simhash_c1gram_dist = calculate_simhash_dist(q1_c, q2_c)
            simhash_c2gram_dist = calculate_simhash_dist(get_ngrams(q1_c, 2), get_ngrams(q2_c, 2))
            simhash_c3gram_dist = calculate_simhash_dist(get_ngrams(q1_c, 3), get_ngrams(q2_c, 3))

            return '{}:{}:{}:{}:{}:{}'.format(simhash_w1gram_dist, simhash_w2gram_dist, simhash_w3gram_dist,
                                              simhash_c1gram_dist, simhash_c2gram_dist, simhash_c3gram_dist)

        df['sim_hash'] = df.apply(calculate_all_simhash, axis=1, raw=True)
        df['simhash_w1gram_dist'] = df['sim_hash'].apply(lambda x: float(x.split(':')[0]))
        df['simhash_w2gram_dist'] = df['sim_hash'].apply(lambda x: float(x.split(':')[1]))
        df['simhash_w3gram_dist'] = df['sim_hash'].apply(lambda x: float(x.split(':')[2]))
        df['simhash_c1gram_dist'] = df['sim_hash'].apply(lambda x: float(x.split(':')[3]))
        df['simhash_c2gram_dist'] = df['sim_hash'].apply(lambda x: float(x.split(':')[4]))
        df['simhash_c3gram_dist'] = df['sim_hash'].apply(lambda x: float(x.split(':')[5]))
        del df['sim_hash']

    def extract_length_feature(self, df):

        def length_compare(row, cpm, w_or_c):
            """
            :param row:
            :param cpm:
            :param type: type in [w, c]
            :return:
            """
            l1 = len(row['q1_{}'.format(w_or_c)].split())
            l2 = len(row['q2_{}'.format(w_or_c)].split())
            return cpm(l1, l2)

        df['len_word_max'] = df.apply(lambda x: length_compare(x, max, 'w'), axis=1, raw=True)
        df['len_word_min'] = df.apply(lambda x: length_compare(x, min, 'w'), axis=1, raw=True)
        df['len_char_max'] = df.apply(lambda x: length_compare(x, max, 'c'), axis=1, raw=True)
        df['len_char_min'] = df.apply(lambda x: length_compare(x, min, 'c'), axis=1, raw=True)

        df['word_length_diff'] = (df['len_word_max'] - df['len_word_min']).abs()
        df['char_length_diff'] = (df['len_char_max'] - df['len_char_min']).abs()

    def extract_distance_feature(self, df):

        def get_vector(df, dictionary):
            q1_vec = [dictionary.doc2bow(s.split()) for s in list(df['q1_w'])]
            q2_vec = [dictionary.doc2bow(s.split()) for s in list(df['q2_w'])]

            q1_vec = gensim.matutils.corpus2csc(q1_vec, num_terms=len(dictionary.token2id))
            q2_vec = gensim.matutils.corpus2csc(q2_vec, num_terms=len(dictionary.token2id))

            return q1_vec.transpose(), q2_vec.transpose()

        def get_similarity_values(q1_csc, q2_csc):
            cosine_sim = []
            manhattan_dis = []
            eucledian_dis = []
            jaccard_dis = []
            minkowsk_dis = []

            for i, j in zip(q1_csc, q2_csc):
                sim = cs(i, j)
                cosine_sim.append(sim[0][0])
                sim = md(i, j)
                manhattan_dis.append(sim[0][0])
                sim = ed(i, j)
                eucledian_dis.append(sim[0][0])
                i_ = i.toarray()
                j_ = j.toarray()
                try:
                    sim = jsc(i_, j_)
                    jaccard_dis.append(sim)
                except:
                    jaccard_dis.append(0)

                sim = minkowski_dis.pairwise(i_, j_)
                minkowsk_dis.append(sim[0][0])
            return cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis

        q1_csc, q2_csc = get_vector(df, self.dictionary)

        cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis = get_similarity_values(q1_csc, q2_csc)
        # print("[FE] cosine_sim sample= \n", cosine_sim[0:2])
        # print("[FE] manhattan_dis sample = \n", manhattan_dis[0:2])
        # print("[FE] eucledian_dis sample = \n", eucledian_dis[0:2])
        # print("[FE] jaccard_dis sample = \n", jaccard_dis[0:2])
        # print("[FE] minkowsk_dis sample = \n", minkowsk_dis[0:2])

        eucledian_dis_array = np.array(eucledian_dis).reshape(-1, 1)
        manhattan_dis_array = np.array(manhattan_dis).reshape(-1, 1)
        minkowsk_dis_array = np.array(minkowsk_dis).reshape(-1, 1)

        eucledian_dis = eucledian_dis_array.flatten()
        manhattan_dis = manhattan_dis_array.flatten()
        minkowsk_dis = minkowsk_dis_array.flatten()

        df['cosine_sim'] = cosine_sim
        df['manhattan_dis'] = manhattan_dis
        df['eucledian_dis'] = eucledian_dis
        df['jaccard_dis'] = jaccard_dis
        df['minkowsk_dis'] = minkowsk_dis

    def extract_word2vec_feature(self, df, w_or_c):
        if w_or_c == 'w':
            df['{}_wmd'.format(w_or_c)] = df.apply(
                lambda row: self.w_word2vec_model.wmdistance(row['q1_{}'.format(w_or_c)].split(),
                                                             row['q2_{}'.format(w_or_c)].split()),
                axis=1)
        else:
            df['{}_wmd'.format(w_or_c)] = df.apply(
                lambda row: self.c_word2vec_model.wmdistance(row['q1_{}'.format(w_or_c)].split(),
                                                             row['q2_{}'.format(w_or_c)].split()),
                axis=1)

    def extract_tfidf_feature(self, df):
        q1_w_vec = self.tfidf_vectorizer.transform(df['q1_w'].values.tolist())
        q2_w_vec = self.tfidf_vectorizer.transform(df['q2_w'].values.tolist())

        df['tfidf_cs'] = np.concatenate([cs(q1_w_vec[i], q2_w_vec[i]).flatten() for i in range(q1_w_vec.shape[0])])
        df['tfidf_ed'] = np.concatenate([ed(q1_w_vec[i], q2_w_vec[i]).flatten() for i in range(q1_w_vec.shape[0])])
        df['tfidf_md'] = np.concatenate([md(q1_w_vec[i], q2_w_vec[i]).flatten() for i in range(q1_w_vec.shape[0])])

        corpus_tfidf = np.concatenate([q1_w_vec.toarray(), q2_w_vec.toarray()], axis=0)

        svd_model = TruncatedSVD(n_components=5)
        svd_model.fit(corpus_tfidf)

        svd_topic = svd_model.transform(corpus_tfidf)
        q1_w_svd_feature = svd_topic[:q1_w_vec.shape[0]]
        q2_w_svd_feature = svd_topic[q1_w_vec.shape[0]:]

        df['svd_cs'] = np.concatenate(
            [cs(q1_w_svd_feature[i].reshape(-1, 5), q2_w_svd_feature[i].reshape(-1, 5)).flatten() for i in
             range(q1_w_svd_feature.shape[0])])
        df['svd_ed'] = np.concatenate(
            [ed(q1_w_svd_feature[i].reshape(-1, 5), q2_w_svd_feature[i].reshape(-1, 5)).flatten() for i in
             range(q1_w_svd_feature.shape[0])])
        df['svd_md'] = np.concatenate(
            [md(q1_w_svd_feature[i].reshape(-1, 5), q2_w_svd_feature[i].reshape(-1, 5)).flatten() for i in
             range(q1_w_svd_feature.shape[0])])

        lda_model = LatentDirichletAllocation(n_components=5, random_state=0)
        lda_model.fit(corpus_tfidf)

        lda_topic = lda_model.transform(corpus_tfidf)

        q1_w_lda_feature = lda_topic[:q1_w_vec.shape[0]]
        q2_w_lda_feature = lda_topic[q1_w_vec.shape[0]:]

        df['lda_cs'] = np.concatenate(
            [cs(q1_w_lda_feature[i].reshape(-1, 5), q2_w_lda_feature[i].reshape(-1, 5)).flatten() for i in
             range(q1_w_lda_feature.shape[0])])
        df['lda_ed'] = np.concatenate(
            [ed(q1_w_lda_feature[i].reshape(-1, 5), q2_w_lda_feature[i].reshape(-1, 5)).flatten() for i in
             range(q1_w_lda_feature.shape[0])])
        df['lda_md'] = np.concatenate(
            [md(q1_w_lda_feature[i].reshape(-1, 5), q2_w_lda_feature[i].reshape(-1, 5)).flatten() for i in
             range(q1_w_lda_feature.shape[0])])

    def extract_magic_feautre(self):
        """
        reference : https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
        :return:
        """
        df1 = self.train_df[['qid1']].copy()
        df2 = self.train_df[['qid2']].copy()

        df1_test = self.test_df[['qid1']].copy()
        df2_test = self.test_df[['qid2']].copy()

        df2.rename(columns={'qid2': 'qid1'}, inplace=True)
        df2_test.rename(columns={'qid2': 'qid1'}, inplace=True)

        train_questions = df1.append(df2)
        train_questions = train_questions.append(df1_test)
        train_questions = train_questions.append(df2_test)
        train_questions.drop_duplicates(subset=['qid1'], inplace=True)

        train_questions.reset_index(drop=True, inplace=True)

        questions_dict = pd.Series(train_questions.index.values, index=train_questions.qid1.values).to_dict()

        train_cp = self.train_df.copy()
        test_cp = self.test_df.copy()
        test_cp['label'] = -1

        # train_cp.drop(['qid1', 'qid2'], inplace=True, axis=1)

        comb = pd.concat([train_cp, test_cp])

        comb['q1_hash'] = comb['qid1'].map(questions_dict)
        comb['q2_hash'] = comb['qid2'].map(questions_dict)

        q1_vc = comb.q1_hash.value_counts().to_dict()
        q2_vc = comb.q2_hash.value_counts().to_dict()

        def try_apply_dict(x, dict_to_apply):
            try:
                return dict_to_apply[x]
            except KeyError:
                return 0

        comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))
        comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))

        comb_train = comb[comb['label'] >= 0]
        comb_test = comb[comb['label'] < 0]

        self.train_df['q1_freq'] = comb_train['q1_freq']
        self.train_df['q2_freq'] = comb_train['q2_freq']

        self.test_df['q1_freq'] = comb_test['q1_freq']
        self.test_df['q2_freq'] = comb_test['q2_freq']

    def extract_graph_feature(self, df):
        def init_graph():
            qs = copy.deepcopy(pd.concat([self.train_df, self.test_df], axis=0))
            g = nx.Graph()
            g.add_nodes_from(qs.qid1)
            g.add_nodes_from(qs.qid2)
            edges = list(df[['qid1', 'qid2']].to_records(index=False))
            g.add_edges_from(edges)
            return g

        def get_edge_count(g, q):
            return g.degree(q)

        def get_edge_count_all(g, row):
            return sum(g.degree([row.qid1, row.qid2]).values)

        def get_edge_count_diff(g, row):
            return g.degree(row.qid1) - g.degree(row.qid2)

        g = init_graph()

        df['q1_degree'] = df.apply(lambda x: get_edge_count(g, x.qid1), axis=1)
        df['q2_degree'] = df.apply(lambda x: get_edge_count(g, x.qid2), axis=1)

    def extract_edit_dist_feature(self, df):
        def edit_dist(a, b):
            """
            Calculates the Levenshtein distance between a and b.
            a and b are both list.
            reference: https://stackoverflow.com/questions/6709693/calculating-the-similarity-of-two-lists
            """
            n, m = len(a), len(b)
            if n > m:
                # Make sure n <= m, to use O(min(n,m)) space
                a, b = b, a
                n, m = m, n
            current = range(n + 1)
            for i in range(1, m + 1):
                previous, current = current, [i] + [0] * n
                for j in range(1, n + 1):
                    add, delete = previous[j] + 1, current[j - 1] + 1
                    change = previous[j - 1]
                    if a[j - 1] != b[i - 1]:
                        change = change + 1
                    current[j] = min(add, delete, change)
                return current[n]

        def compute_edit_dist(row, w_or_c):
            q1 = row['q1_{}'.format(w_or_c)].split()
            q2 = row['q2_{}'.format(w_or_c)].split()
            return edit_dist(q1, q2)

        df['edit_dist_word'] = df.apply(lambda x: compute_edit_dist(x, w_or_c='w'), axis=1)
        df['edit_dist_char'] = df.apply(lambda x: compute_edit_dist(x, w_or_c='c'), axis=1)

        pass

    def extract_jaccard_dist_feature(self, df):

        pass

    def extract_intersection_feature(self, df):
        def inter_word_num(row, w_or_c):
            q1 = row['q1_{}'.format(w_or_c)].split()
            q2 = row['q2_{}'.format(w_or_c)].split()

            interaction = set(q1).intersection(set(q2))

            return len(interaction)

        df['w_intersection'] = df.apply(lambda x: inter_word_num(x, 'w'), axis=1)
        df['c_intersection'] = df.apply(lambda x: inter_word_num(x, 'c'), axis=1)

    def extract_share_feature(self, df):

        def w_or_c_match_share(row, w_or_c):
            q1words = {}
            q2words = {}
            for word in row['q1_{}'.format(w_or_c)].split():
                q1words[word] = 1
            for word in row['q2_{}'.format(w_or_c)].split():
                q2words[word] = 1

            if len(q1words) == 0 or len(q2words) == 0:
                # The computer-generated chaff includes a few questions that are nothing but stopwords
                return 0
            shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
            shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
            R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
            return R

        def get_weight(count, eps=10000, min_count=2):
            if count < min_count:
                return 0
            else:
                return 1 / (count + eps)

        def load_word_weight():
            train_qs = self.train_df['q1_w'].tolist() + self.train_df['q2_w'].tolist()
            # train_qs += pd.Series(self.test_df['q1_w'].tolist() + self.test_df['q2_w'].tolist())
            train_qs = [x.split() for x in train_qs]
            words = [x for y in train_qs for x in y]
            counts = Counter(words)
            weights = {word: get_weight(count) for word, count in counts.items()}
            return weights

        def load_char_weight():
            train_qs = self.train_df['q1_c'].tolist() + self.train_df['q2_c'].tolist()
            # train_qs += pd.Series(self.test_df['q1_c'].tolist() + self.test_df['q2_c'].tolist())
            train_qs = [x.split() for x in train_qs]
            words = [x for y in train_qs for x in y]
            counts = Counter(words)
            weights = {word: get_weight(count) for word, count in counts.items()}
            return weights

        def tfidf_word_match_share(row, weights=None):
            q1words = {}
            q2words = {}
            for word in row['q1_w'].split():
                q1words[word] = 1
            for word in row['q2_w'].split():
                q2words[word] = 1
            if len(q1words) == 0 or len(q2words) == 0:
                # The computer-generated chaff includes a few questions that are nothing but stopwords
                return 0

            shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                            q2words.keys() if
                                                                                            w in q1words]
            total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

            R = np.sum(shared_weights) / np.sum(total_weights)
            return R

        def tfidf_char_match_share(row, weights=None):
            q1words = {}
            q2words = {}
            for word in row['q1_c'].split():
                q1words[word] = 1
            for word in row['q2_c'].split():
                q2words[word] = 1
            if len(q1words) == 0 or len(q2words) == 0:
                # The computer-generated chaff includes a few questions that are nothing but stopwords
                return 0

            shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                            q2words.keys() if
                                                                                            w in q1words]
            total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

            R = np.sum(shared_weights) / np.sum(total_weights)
            return R

        df['word_match_share'] = df.apply(lambda x: w_or_c_match_share(x, w_or_c='w'), axis=1)
        df['char_match_share'] = df.apply(lambda x: w_or_c_match_share(x, w_or_c='c'), axis=1)

        word_weight = load_word_weight()
        char_weight = load_char_weight()

        df['tfidf_word_match_share'] = df.apply(lambda x: tfidf_word_match_share(x, weights=word_weight), axis=1)
        df['tfidf_char_match_share'] = df.apply(lambda x: tfidf_char_match_share(x, weights=char_weight), axis=1)

        pass

    def extract_fuzzywuzzy_feature(self, df):
        df['token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(x['q1_w'], x['q2_w']), axis=1)
        df['token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(x['q1_w'], x['q2_w']), axis=1)
        df['fuzz_ratio'] = df.apply(lambda x: fuzz.QRatio(x['q1_w'], x['q2_w']), axis=1)
        df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(x['q1_w'], x['q2_w']), axis=1)

        df['token_set_ratio_char'] = df.apply(lambda x: fuzz.token_set_ratio(x['q1_c'], x['q2_c']), axis=1)
        df['token_sort_ratio_char'] = df.apply(lambda x: fuzz.token_sort_ratio(x['q1_c'], x['q2_c']), axis=1)
        df['fuzz_ratio_char'] = df.apply(lambda x: fuzz.QRatio(x['q1_c'], x['q2_c']), axis=1)
        df['fuzz_partial_ratio_char'] = df.apply(lambda x: fuzz.partial_ratio(x['q1_c'], x['q2_c']), axis=1)

    def extract_bm25_feature(self, df):
        """
        非常耗时，3个半小时
        :param df:
        :return:
        """
        def bm25_dist(row, dist_type, bm25_model, average_idf, feature_dim):
            assert dist_type in ['cs', 'ed', 'md'], 'dist type error'
            q1 = row['q1_w'].split()
            q2 = row['q2_w'].split()
            q1_bm25 = bm25_model.get_scores(q1, average_idf)
            q2_bm25 = bm25_model.get_scores(q2, average_idf)
            q1_bm25 = np.reshape(np.array(q1_bm25), (-1, feature_dim))
            q2_bm25 = np.reshape(np.array(q2_bm25), (-1, feature_dim))

            if dist_type == 'cs':
                score = cs(q1_bm25, q2_bm25).flatten()[0]
            elif dist_type == 'ed':
                score = ed(q1_bm25, q2_bm25).flatten()[0]
            elif dist_type == 'md':
                score = md(q1_bm25, q2_bm25).flatten()[0]
            return score

        df['bm25_cs'] = df.apply(
            lambda row: bm25_dist(row, dist_type='cs', bm25_model=self.bm25_model, average_idf=self.bm25_avg_idf,
                                  feature_dim=self.bm25_dim), axis=1)

        df['bm25_ed'] = df.apply(
            lambda row: bm25_dist(row, dist_type='ed', bm25_model=self.bm25_model, average_idf=self.bm25_avg_idf,
                                  feature_dim=self.bm25_dim), axis=1)

        df['bm25_md'] = df.apply(
            lambda row: bm25_dist(row, dist_type='md', bm25_model=self.bm25_model, average_idf=self.bm25_avg_idf,
                                  feature_dim=self.bm25_dim), axis=1)

    def extract_word_pair_feature(self, df):
        def init_word_pair_graph():
            g = nx.Graph()
            w_list = []

            q1s = list(self.train_df['q1_w'].values)
            q2s = list(self.train_df['q2_w'].values)

            for q in q1s:
                w_list += q.split()

            for q in q2s:
                w_list += q.split()

            g.add_nodes_from(w_list)

            for idx in range(len(self.train_df)):
                q1 = self.train_df.loc[idx, 'q1_w'].split()
                q2 = self.train_df.loc[idx, 'q2_w'].split()
                label = self.train_df.loc[idx, 'label']
                for w1 in q1:
                    for w2 in q2:
                        if w1 != w2:
                            g.add_edge(w1, w2)
                            if label in g[w1][w2]:
                                g[w1][w2][label] += 1
                            else:
                                g[w1][w2][label] = 1
            return g

        def get_label_count(row, g):
            q1 = row['q1_w'].split()
            q2 = row['q2_w'].split()
            label_cnt = {}
            label_cnt[0] = 0
            label_cnt[1] = 0
            for w1 in q1:
                for w2 in q2:
                    try:
                        edge = g[w1][w2]
                        label_cnt[0] += edge[0]
                        label_cnt[1] += edge[1]
                    except:
                        continue
            return '{}:{}'.format(label_cnt[0], label_cnt[1])

        g = init_word_pair_graph()

        df['label_cnt_tmp'] = df.apply(lambda x: get_label_count(x, g), axis=1)
        df['label_cnt_0'] = df['label_cnt_tmp'].apply(lambda x: float(x.split(':')[0]))
        df['label_cnt_1'] = df['label_cnt_tmp'].apply(lambda x: float(x.split(':')[1]))
        del df['label_cnt_tmp']
        pass


def load_data():
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    questions = pd.read_csv('../data/question_id.csv')

    def get_sequence(df, q='q1', left_id='qid1'):
        tmp = pd.merge(df, questions, left_on=left_id, right_on='qid', how='left')
        df['{}_w'.format(q)] = tmp['wid'].values
        df['{}_c'.format(q)] = tmp['cid'].values

    get_sequence(train_data, q='q1', left_id='qid1')
    get_sequence(test_data, q='q1', left_id='qid1')

    get_sequence(train_data, q='q2', left_id='qid2')
    get_sequence(test_data, q='q2', left_id='qid2')

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    return train_data, test_data


def show_feature():
    train_feature = pd.read_csv('../feature/train_feature.csv')
    print(train_feature.columns)


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # show_feature()

    train_data, test_data = load_data()

    # def init_bm25():
    #     questions = pd.read_csv('../data/question_id.csv')
    #
    #     corpus = questions.wid.tolist()
    #
    #     corpus = [s.split() for s in corpus]
    #     corpus = corpus[:1000]
    #     print(corpus[:3])
    #     from gensim.summarization.bm25 import get_bm25_weights, BM25
    #     bm25_model = BM25(corpus)
    #     average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
    #     q1 = bm25_model.get_scores(corpus[1], average_idf)
    #     q2 = bm25_model.get_scores(corpus[0], average_idf)
    #     q1 = np.array(q1)
    #     q2 = np.array(q2)
    #     feature_dim = len(q1)
    #     q1 = np.reshape(q1, (-1, feature_dim))
    #     q2 = np.reshape(q2, (-1, feature_dim))
    #     print(type(q1))
    #     print(q1.shape)
    #     print(cs(q1, q2).flatten())
    #
    # init_bm25()

    fe = FeatureExtractor(train_df=train_data, test_df=test_data, w_embed_file='../data/word_embed_v1.txt',
                          c_embed_file='../data/char_embed_v1.txt')

    train_data, test_data = fe.extract_all_feature()

    train_data.to_csv('../feature/train_feature_wp.csv', index=False)
    test_data.to_csv('../feature/test_feature_wp.csv', index=False)
    pass
