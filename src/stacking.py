import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import copy
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import catboost
import xgboost
from scipy.sparse import csc_matrix


def ensemble_rf(train_probs, test_probs, train_label, use_feature=False):
    # 随机森林分类器
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1
        # print('RF fold [{}]'.format(split_id))
        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = RandomForestClassifier()
        model.fit(t_probs, t_label)
        v_pred = model.predict(v_probs)
        # train_pred.append(v_pred)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(v_probs)
        train_pred.append(tmp)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    # predict result on train data
    train_pred = np.concatenate(train_pred, axis=0)
    train_pred = train_pred[:, 1]

    # predict result on test data
    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp[:, 1] / 5.0

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_lr(train_probs, test_probs, train_label, use_feature=False):
    from sklearn.linear_model import LogisticRegression
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1
        # print('LR fold [{}]'.format(split_id))
        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = LogisticRegression()
        model.fit(t_probs, t_label)

        v_pred = model.predict(v_probs)
        # train_pred.append(v_pred)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(v_probs)
        train_pred.append(tmp)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)
    train_pred = train_pred[:, 1]

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp[:, 1] / 5.0
    return total_f1 / 5.0, train_pred, test_pred


def ensemble_dt(train_probs, test_probs, train_label, use_feature=False):
    from sklearn.tree import DecisionTreeClassifier
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1
        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = DecisionTreeClassifier()
        model.fit(t_probs, t_label)

        v_pred = model.predict(v_probs)
        train_pred.append(v_pred)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp.argmax(axis=1)

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_ada(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1
        # print('LR fold [{}]'.format(split_id))
        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = AdaBoostClassifier(n_estimators=60, random_state=2018, learning_rate=0.01)

        model.fit(t_probs, t_label)
        v_pred = model.predict(v_probs)
        # train_pred.append(v_pred)
        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(v_probs)
        train_pred.append(tmp)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)
    train_pred = train_pred[:, 1]

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp[:, 1] / 5.0

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_gb(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = GradientBoostingClassifier(learning_rate=0.06, n_estimators=100, subsample=0.8, random_state=2017,
                                           max_depth=5, verbose=0)
        model.fit(t_probs, t_label)

        v_pred = model.predict(v_probs)
        # train_pred.append(v_pred)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(v_probs)
        train_pred.append(tmp)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)
    train_pred = train_pred[:, 1]

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp[:, 1] / 5.0

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_gnb(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = GaussianNB()
        model.fit(t_probs, t_label)

        v_pred = model.predict(v_probs)
        # train_pred.append(v_pred)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(v_probs)
        train_pred.append(tmp)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)
    train_pred = train_pred[:, 1]

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp[:, 1] / 5.0

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_nn(train_probs, test_probs, train_label, use_feature=False):
    def get_model(feature_dim):
        pass

    pass


def ensemble_fm(train_probs, test_probs, train_label, use_feature=False):
    from fastFM import sgd
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    test_probs = csc_matrix(test_probs)

    train_label = copy.deepcopy(train_label)
    train_label = np.where(train_label == 0, -1, 1)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        t_probs = copy.deepcopy(csc_matrix(t_probs, dtype=np.float))
        v_probs = copy.deepcopy(csc_matrix(v_probs, dtype=np.float))

        model = sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2, step_size=0.1)
        model.fit(t_probs, t_label)

        v_pred = model.predict(v_probs)
        v_pred = np.where(v_pred < 0, 0, 1)
        train_pred.append(v_pred)
        v_label = copy.deepcopy(np.where(v_label < 0, 0, 1))
        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict(test_probs)
        tmp = tmp[:, np.newaxis]
        tmp = np.where(tmp < 0, 0, 1)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)

    test_pred = np.concatenate(test_pred, axis=1)
    test_pred = np.sum(test_pred, axis=1)
    test_pred = np.where(test_pred >= 3, 1, 0)

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_ffm(train_probs, test_probs, train_label, use_feature=False):
    pass


def ensemble_lsvc(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0
    train_pred = []
    test_pred = []
    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = LinearSVC(random_state=2017)
        model.fit(t_probs, t_label)

        v_pred = model.predict(v_probs)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict(v_probs)
        train_pred.append(tmp)

        tmp = model.predict(test_probs)
        tmp = tmp[:, np.newaxis]
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)

    test_pred = np.concatenate(test_pred, axis=1)
    test_pred = np.sum(test_pred, axis=1)
    test_pred = np.where(test_pred >= 3, 1, 0)

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_et(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = ExtraTreesClassifier(n_estimators=1200, max_depth=24, max_features="auto", n_jobs=-1, random_state=2017,
                                     verbose=0)
        model.fit(t_probs, t_label)

        v_pred = model.predict(v_probs)
        # train_pred.append(v_pred)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(v_probs)
        train_pred.append(tmp)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)
    train_pred = train_pred[:, 1]

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp[:, 1] / 5.0

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_cat(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = catboost.CatBoostClassifier(loss_function='Logloss',
                                            eval_metric='AUC',
                                            iterations=5000,
                                            learning_rate=0.02,
                                            depth=6,
                                            rsm=0.7,
                                            od_type='Iter',
                                            od_wait=700,
                                            logging_level='Silent',
                                            allow_writing_files=False,
                                            metric_period=100,
                                            random_seed=1)

        model.fit(t_probs, t_label, eval_set=(v_probs, v_label), use_best_model=True)

        v_pred = model.predict(v_probs)
        # train_pred.append(v_pred)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(v_probs)
        train_pred.append(tmp)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)
    train_pred = train_pred[:, 1]

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp[:, 1] / 5.0

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_lgbm(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_feature = pd.read_csv('../feature/train_feature.csv')
        train_feature = train_feature.values
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.concatenate([train_probs, train_feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    total_f1 = 0.0
    kf = KFold(n_splits=5)
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'min_child_weight': 1.5,
            'num_leaves': 2 ** 5,
            'lambda_l2': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'colsample_bylevel': 0.7,
            'learning_rate': 0.01,
            'seed': 2017,
            'nthread': 12,
            'silent': True,
            'verbose': -1
        }
        num_round = 10000
        early_stopping_rounds = 100

        lgb_train_data = lgb.Dataset(t_probs, t_label)
        lgb_val_data = lgb.Dataset(v_probs, v_label)

        lgbm = lgb.train(params=params, train_set=lgb_train_data, num_boost_round=num_round, valid_sets=lgb_val_data,
                         early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

        v_pred = lgbm.predict(v_probs, num_iteration=lgbm.best_iteration)
        v_pred = np.where(v_pred > 0.5, 1, 0)

        total_f1 += f1_score(v_label, v_pred)
        tmp = lgbm.predict(v_probs)
        train_pred.append(tmp)

        tmp = lgbm.predict(test_probs, num_iteration=lgbm.best_iteration)
        tmp = tmp[:, np.newaxis]
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)

    test_pred = np.concatenate(test_pred, axis=1)
    test_pred = np.mean(test_pred, axis=1)

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_xgb(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    test_probs = xgboost.DMatrix(test_probs)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'gamma': 1,
                  'min_child_weight': 1.5,
                  'max_depth': 7,
                  'lambda': 10,
                  'subsample': 0.7,
                  'colsample_bytree': 0.7,
                  'colsample_bylevel': 0.7,
                  'eta': 0.02,
                  'tree_method': 'exact',
                  'seed': 2018,
                  'nthread': 12,
                  'silent': 1
                  # "num_class": 2
                  }

        num_round = 10000
        early_stopping_rounds = 100

        train_matrix = xgboost.DMatrix(t_probs, t_label)
        val_matrix = xgboost.DMatrix(v_probs, v_label)

        watchlist = [(train_matrix, 'train'),
                     (val_matrix, 'eval')]

        model = xgboost.train(params,
                              train_matrix,
                              num_round,
                              evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=False
                              # verbose=2
                              )

        v_pred = model.predict(val_matrix, )
        v_pred = np.where(v_pred > 0.5, 1, 0)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict(val_matrix)
        train_pred.append(tmp)

        tmp = model.predict(test_probs)
        tmp = tmp[:, np.newaxis]
        test_pred.append(tmp)
    train_pred = np.concatenate(train_pred, axis=0)
    test_pred = np.concatenate(test_pred, axis=1)
    test_pred = np.mean(test_pred, axis=1)
    return total_f1 / 5.0, train_pred, test_pred


def ensemble_knn(train_probs, test_probs, train_label, use_feature=False):
    if use_feature is True:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/train_feature.csv')
        feature = feature.values
        train_probs = np.concatenate([train_probs, feature], axis=1)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)
        feature = pd.read_csv('../feature/test_feature.csv')
        feature = feature.values
        test_probs = np.concatenate([test_probs, feature], axis=1)
    else:
        train_probs = copy.deepcopy(train_probs)
        train_probs = np.where(train_probs > 0.5, 1, 0)

        test_probs = copy.deepcopy(test_probs)
        test_probs = np.where(test_probs > 0.5, 1, 0)

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = KNeighborsClassifier(n_neighbors=200, n_jobs=-1)
        model.fit(t_probs, t_label)

        v_pred = model.predict(v_probs)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(v_probs)
        train_pred.append(tmp)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)
    train_pred = train_pred[:, 1]

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp[:, 1] / 5.0

    return total_f1 / 5.0, train_pred, test_pred


def ensemble_avg(train_probs, test_probs, train_label):
    train_pred = copy.deepcopy(np.mean(train_probs, axis=1))
    train_pred = np.where(train_pred > 0.5, 1, 0)
    f1 = f1_score(train_label, train_pred)

    train_pred = copy.deepcopy(np.mean(train_probs, axis=1))
    test_pred = np.mean(test_probs, axis=1)

    return f1, train_pred, test_pred


def stacking_layer_2(use_feature=True):
    """
    第二层stackig
    :return:
    """

    layer2_train_in = []
    layer2_test_in = []

    questions = pd.read_csv('../data/question_id.csv')
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    train_label = train_data['label'].values

    model_name = ['esim_char',
                  'esim_word',
                  'bimpm_char',
                  'bimpm_word',
                  'esim_char_gl_feature',
                  'esim_word_gl_feature',
                  'dcnn_char_gl_feature',
                  'dcnn_word_gl_feature',
                  # 'embed_cnn_char',
                  # 'embed_cnn_word',
                  'my_nn_char',
                  'my_nn_word'
                  ]

    train_result = []
    test_result = []
    for model in model_name:  #
        for round_id in range(1, 6):
            train_probs = []
            test_probs = []
            for i in range(1, 6):
                tmp = np.load('../prediction/{}/round_{}_train_kf_{}.npy'.format(model, round_id, i))
                train_probs.append(tmp)

                tmp = np.load('../prediction/{}/round_{}_test_kf_{}.npy'.format(model, round_id, i))
                test_probs.append(tmp)
            train_probs = np.concatenate(train_probs, axis=0)
            train_result.append(train_probs)

            test_probs = np.concatenate(test_probs, axis=1)
            test_probs = np.mean(test_probs, axis=1)[:, np.newaxis]
            test_result.append(test_probs)

    train_probs = np.concatenate(train_result, axis=1)
    test_probs = np.concatenate(test_result, axis=1)

    print(train_probs.shape)
    print(test_probs.shape)

    print('Ensemble Average f1:')
    tmp = ensemble_avg(train_probs, test_probs, train_label)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble RandomForest f1(no feature):')
    tmp = ensemble_rf(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble RandomForest f1(feature):')
    tmp = ensemble_rf(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble LogisticRegression f1(no feature):')
    tmp = ensemble_lr(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble LogisticRegression f1(feature):')
    tmp = ensemble_lr(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    # print('Ensemble DecisionTree f1(no feature):')
    # tmp = ensemble_dt(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble DecisionTree f1(feature):')
    # tmp = ensemble_dt(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)

    print('Ensemble adaboost f1(no feature):')
    tmp = ensemble_ada(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble adaboost f1(feature):')
    tmp = ensemble_ada(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble gb f1(no feature):')
    tmp = ensemble_gb(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble gb f1(feature):')
    tmp = ensemble_gb(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble GaussianNB f1(no feature):')
    tmp = ensemble_gnb(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble GaussianNB f1(feature):')
    tmp = ensemble_gnb(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble LinearSVC f1(no feature):')
    tmp = ensemble_lsvc(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble LinearSVC f1(feature):')
    tmp = ensemble_lsvc(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble ExtraTreesClassifier f1(no feature):')
    tmp = ensemble_et(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble ExtraTreesClassifier f1(feature):')
    tmp = ensemble_et(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble catboost f1(no feature):')
    tmp = ensemble_cat(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble catboost f1(feature):')
    tmp = ensemble_cat(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble xgbboost f1(no feature):')
    tmp = ensemble_xgb(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble xgbboost f1(feature):')
    tmp = ensemble_xgb(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble lightgbm f1(no feature):')
    tmp = ensemble_lgbm(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble lightgbm f1(feature):')
    tmp = ensemble_lgbm(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble knn f1(no feature):')
    tmp = ensemble_knn(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    print('Ensemble knn f1(feature):')
    tmp = ensemble_knn(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)

    layer2_train_in.append(tmp[1])
    layer2_test_in.append(tmp[2])

    for idx, tmp in enumerate(layer2_train_in):
        tmp = tmp[:, np.newaxis]
        layer2_train_in[idx] = tmp

    for idx, tmp in enumerate(layer2_test_in):
        tmp = tmp[:, np.newaxis]
        layer2_test_in[idx] = tmp

    layer2_train_feature = np.concatenate(layer2_train_in, axis=1)
    layer2_test_feature = np.concatenate(layer2_test_in, axis=1)

    print('#################################')
    print('#################################')
    print('           Stacking Layer 2      ')
    print('#################################')
    print('#################################')

    print(layer2_train_feature.shape)

    np.save('../stacking_2layers/train_labels.npy', layer2_train_feature)
    np.save('../stacking_2layers/test_labels.npy', layer2_test_feature)

    # if use_feature is True:

    train_feature = pd.read_csv('../feature/train_feature.csv')
    train_feature = train_feature.values

    layer2_train_feature = np.concatenate([layer2_train_feature, train_feature], axis=1)

    test_feature = pd.read_csv('../feature/test_feature.csv')
    test_feature = test_feature.values

    layer2_test_feature = np.concatenate([layer2_test_feature, test_feature], axis=1)

    kf = KFold(n_splits=5)
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(layer2_train_feature):
        split_id += 1

        t_probs = layer2_train_feature[t_idx]
        v_probs = layer2_train_feature[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'min_child_weight': 1.5,
            'num_leaves': 2 ** 5,
            'lambda_l2': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'colsample_bylevel': 0.7,
            'learning_rate': 0.01,
            'seed': 2017,
            'nthread': 12,
            'silent': True,
            'verbose': -1
        }
        num_round = 10000
        early_stopping_rounds = 100

        lgb_train_data = lgb.Dataset(t_probs, t_label)
        lgb_val_data = lgb.Dataset(v_probs, v_label)

        lgbm = lgb.train(params=params, train_set=lgb_train_data, num_boost_round=num_round,
                         valid_sets=lgb_val_data,
                         early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

        v_pred = lgbm.predict(v_probs, num_iteration=lgbm.best_iteration)
        v_pred = np.where(v_pred > 0.5, 1, 0)
        train_pred.append(v_pred)

        tmp = lgbm.predict(layer2_test_feature, num_iteration=lgbm.best_iteration)
        tmp = tmp[:, np.newaxis]
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)

    test_pred = np.concatenate(test_pred, axis=1)
    test_pred = np.mean(test_pred, axis=1)
    test_pred = np.where(test_pred > 0.5, 1, 0)
    print(test_pred[:10])

    print(f1_score(train_label, train_pred))




    # print('Ensemble FM f1(no feature):')
    # tmp = ensemble_fm(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble FM f1(feature):')
    # tmp = ensemble_fm(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)


def test_classifier():
    questions = pd.read_csv('../data/question_id.csv')
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    train_label = train_data['label'].values

    model_name = ['esim_char',
                  'esim_word',
                  'bimpm_char',
                  'bimpm_word',
                  'esim_char_gl_feature',
                  'esim_word_gl_feature',
                  'dcnn_char_gl_feature',
                  'dcnn_word_gl_feature',
                  # 'de_att_word',
                  # 'de_att_char',
                  # 'embed_cnn_char',
                  # 'embed_cnn_word',
                  'my_nn_char',
                  'my_nn_word'
                  ]

    train_result = []
    test_result = []
    for model in model_name:
        for round_id in range(1, 6):
            train_probs = []
            test_probs = []
            for i in range(1, 6):
                tmp = np.load('../prediction/{}/round_{}_train_kf_{}.npy'.format(model, round_id, i))
                train_probs.append(tmp)

                tmp = np.load('../prediction/{}/round_{}_test_kf_{}.npy'.format(model, round_id, i))
                test_probs.append(tmp)
            train_probs = np.concatenate(train_probs, axis=0)
            train_result.append(train_probs)

            test_probs = np.concatenate(test_probs, axis=1)
            test_probs = np.mean(test_probs, axis=1)[:, np.newaxis]
            test_result.append(test_probs)

    train_probs = np.concatenate(train_result, axis=1)
    test_probs = np.concatenate(test_result, axis=1)

    print(train_probs.shape)
    print(test_probs.shape)

    # print('Ensemble Average f1:')
    # tmp = ensemble_avg(train_probs, test_probs, train_label)
    # print(tmp)
    #
    # print('Ensemble RandomForest f1(no feature):')
    # tmp = ensemble_rf(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble RandomForest f1(feature):')
    # tmp = ensemble_rf(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)
    #
    # print('Ensemble LogisticRegression f1(no feature):')
    # tmp = ensemble_lr(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble LogisticRegression f1(feature):')
    # tmp = ensemble_lr(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)

    # print('Ensemble adaboost f1(no feature):')
    # tmp = ensemble_ada(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble adaboost f1(feature):')
    # tmp = ensemble_ada(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)

    # print('Ensemble gb f1(no feature):')
    # tmp = ensemble_gb(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble gb f1(feature):')
    # tmp = ensemble_gb(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)

    # print('Ensemble GaussianNB f1(no feature):')
    # tmp = ensemble_gnb(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble GaussianNB f1(feature):')
    # tmp = ensemble_gnb(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)
    #
    # print('Ensemble LinearSVC f1(no feature):')
    # tmp = ensemble_lsvc(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble LinearSVC f1(feature):')
    # tmp = ensemble_lsvc(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)
    #
    # print('Ensemble ExtraTreesClassifier f1(no feature):')
    # tmp = ensemble_et(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble ExtraTreesClassifier f1(feature):')
    # tmp = ensemble_et(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)
    #
    # print('Ensemble catboost f1(no feature):')
    # tmp = ensemble_cat(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    # #
    # print('Ensemble catboost f1(feature):')
    # tmp = ensemble_cat(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)
    #
    # print('Ensemble xgbboost f1(no feature):')
    # tmp = ensemble_xgb(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble xgbboost f1(feature):')
    # tmp = ensemble_xgb(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)
    #
    # print('Ensemble lightgbm f1(no feature):')
    # tmp = ensemble_lgbm(train_probs, test_probs, train_label, use_feature=False)
    # print(tmp)
    #
    # print('Ensemble lightgbm f1(feature):')
    # tmp = ensemble_lgbm(train_probs, test_probs, train_label, use_feature=True)
    # print(tmp)
    #
    print('Ensemble knn f1(no feature):')
    tmp = ensemble_knn(train_probs, test_probs, train_label, use_feature=False)
    print(tmp)

    print('Ensemble knn f1(feature):')
    tmp = ensemble_knn(train_probs, test_probs, train_label, use_feature=True)
    print(tmp)



def average_stacking_result():


    questions = pd.read_csv('../data/question_id.csv')
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    train_label = train_data['label'].values

    model_name = [
                  'esim_char',
                  'esim_word',
                  'bimpm_char',
                  'bimpm_word',
                  'esim_char_gl_feature',
                  'esim_word_gl_feature',
                  # 'dcnn_char_gl_feature',
                  # 'dcnn_word_gl_feature',
                  # 'de_att_word',
                  # 'de_att_char',
                  # 'embed_cnn_char',
                  # 'embed_cnn_word',
                  'my_nn_char',
                  'my_nn_word'
                  ]

    train_result = []
    test_result = []
    for model in model_name:
        for round_id in range(1, 6):
            train_probs = []
            test_probs = []
            for i in range(1, 6):
                tmp = np.load('../prediction/{}/round_{}_train_kf_{}.npy'.format(model, round_id, i))
                train_probs.append(tmp)

                tmp = np.load('../prediction/{}/round_{}_test_kf_{}.npy'.format(model, round_id, i))
                test_probs.append(tmp)
            train_probs = np.concatenate(train_probs, axis=0)
            train_result.append(train_probs)

            test_probs = np.concatenate(test_probs, axis=1)
            test_probs = np.mean(test_probs, axis=1)[:, np.newaxis]
            test_result.append(test_probs)

    train_probs = np.concatenate(train_result, axis=1)
    test_probs = np.concatenate(test_result, axis=1)

    print(train_probs.shape)
    print(test_probs.shape)

    for i in range(1):
        print('############################')
        print('#####Stacking level {}######'.format(i))
        print('############################')


        train_stacking_result = []
        test_stacking_result = []
        print('Ensemble Average f1:')
        tmp = ensemble_avg(train_probs, test_probs, train_label)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble RandomForest f1(no feature):')
        tmp = ensemble_rf(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble RandomForest f1(feature):')
        tmp = ensemble_rf(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble LogisticRegression f1(no feature):')
        tmp = ensemble_lr(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble LogisticRegression f1(feature):')
        tmp = ensemble_lr(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble adaboost f1(no feature):')
        tmp = ensemble_ada(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble adaboost f1(feature):')
        tmp = ensemble_ada(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble gb f1(no feature):')
        tmp = ensemble_gb(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble gb f1(feature):')
        tmp = ensemble_gb(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble GaussianNB f1(no feature):')
        tmp = ensemble_gnb(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble GaussianNB f1(feature):')
        tmp = ensemble_gnb(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble LinearSVC f1(no feature):')
        tmp = ensemble_lsvc(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble LinearSVC f1(feature):')
        tmp = ensemble_lsvc(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble ExtraTreesClassifier f1(no feature):')
        tmp = ensemble_et(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble ExtraTreesClassifier f1(feature):')
        tmp = ensemble_et(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble catboost f1(no feature):')
        tmp = ensemble_cat(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)
        test_pred = tmp[-1]
        test_pred = np.where(test_pred > 0.5, 1, 0)
        test_data['label'] = test_pred
        test_data.to_csv('../submit/final_v4_cat_nf_{}.csv'.format(i), index=False)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])
        #
        print('Ensemble catboost f1(feature):')
        tmp = ensemble_cat(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)
        test_pred = tmp[-1]
        test_pred = np.where(test_pred > 0.5, 1, 0)
        test_data['label'] = test_pred
        test_data.to_csv('../submit/final_v4_cat_f_{}.csv'.format(i), index=False)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble xgbboost f1(no feature):')
        tmp = ensemble_xgb(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble xgbboost f1(feature):')
        tmp = ensemble_xgb(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        test_pred = tmp[-1]
        test_pred = np.where(test_pred > 0.5, 1, 0)
        test_data['label'] = test_pred
        test_data.to_csv('../submit/final_v5_xgb_f_{}.csv'.format(i), index=False)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble lightgbm f1(no feature):')
        tmp = ensemble_lgbm(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble lightgbm f1(feature):')
        tmp = ensemble_lgbm(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble knn f1(no feature):')
        tmp = ensemble_knn(train_probs, test_probs, train_label, use_feature=False)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print('Ensemble knn f1(feature):')
        tmp = ensemble_knn(train_probs, test_probs, train_label, use_feature=True)
        print(tmp)

        train_stacking_result.append(tmp[1])
        test_stacking_result.append(tmp[2])

        print(len(train_stacking_result))
        print(len(test_stacking_result))

        print('Train item length:')
        for idx, item in enumerate(train_stacking_result):
            item = item[:, np.newaxis]
            train_stacking_result[idx] = item
            # print(item.shape)

        train_probs = np.concatenate(train_stacking_result, axis=1)
        # train_pred = np.mean(train_probs, axis=1)
        # train_pred = np.where(train_pred > 0.5, 1, 0)
        # print('F1 score : {}'.format(f1_score(train_label, train_pred)))

        print('Test item length:')
        for idx, item in enumerate(test_stacking_result):
            item = item[:, np.newaxis]
            test_stacking_result[idx] = item
            # print(item.shape)

        test_probs = np.concatenate(test_stacking_result, axis=1)
        # test_pred = np.mean(test_probs, axis=1)
        # test_pred = np.where(test_pred > 0.1, 1, 0)
        # test_data['label'] = test_pred
        # test_data.to_csv('../submit/final_v3.csv', index=False)




def test():
    questions = pd.read_csv('../data/question_id.csv')
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    train_label = train_data['label'].values

    layer2_train_feature = np.load('../stacking_2layers/train_labels.npy')
    layer2_test_feature = np.load('../stacking_2layers/test_labels.npy')

    train_feature = pd.read_csv('../feature/train_feature.csv')
    train_feature = train_feature.values

    layer2_train_feature = np.concatenate([layer2_train_feature, train_feature], axis=1)

    test_feature = pd.read_csv('../feature/test_feature.csv')
    test_feature = test_feature.values

    layer2_test_feature = np.concatenate([layer2_test_feature, test_feature], axis=1)

    train_probs = layer2_train_feature
    test_probs = layer2_test_feature

    kf = KFold(n_splits=5)
    total_f1 = 0.0
    split_id = 0

    train_pred = []
    test_pred = []

    for t_idx, v_idx in kf.split(train_probs):
        split_id += 1

        t_probs = train_probs[t_idx]
        v_probs = train_probs[v_idx]

        t_label = train_label[t_idx]
        v_label = train_label[v_idx]

        model = catboost.CatBoostClassifier(loss_function='Logloss',
                                            eval_metric='AUC',
                                            iterations=5000,
                                            learning_rate=0.02,
                                            depth=6,
                                            rsm=0.7,
                                            od_type='Iter',
                                            od_wait=700,
                                            logging_level='Silent',
                                            allow_writing_files=False,
                                            metric_period=100,
                                            random_seed=1)

        model.fit(t_probs, t_label, eval_set=(v_probs, v_label), use_best_model=True)

        v_pred = model.predict(v_probs)
        train_pred.append(v_pred)

        total_f1 += f1_score(v_label, v_pred)

        tmp = model.predict_proba(test_probs)
        test_pred.append(tmp)

    train_pred = np.concatenate(train_pred, axis=0)

    tmp = np.zeros_like(test_pred[0])
    for item in test_pred:
        tmp += item
    test_pred = tmp.argmax(axis=1)
    print(test_pred[:10])

    print(f1_score(train_label, train_pred))


if __name__ == '__main__':
    # stacking_layer_2()
    # test_classifier()
    # test()
    average_stacking_result()
