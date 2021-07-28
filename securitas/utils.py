from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pickle
import sys
from time import time


def p_log(*ks, **kwargs):
    print(*ks, **kwargs)
    sys.stdout.flush()


def construct_traffic(filename, LABELS):
    X = [[] for i in range(len(LABELS))]
    with open(filename) as f:
        for i in f.readlines():
            i = i.strip().split()
            tag = i[0].split('//')[0]
            if tag in LABELS:
                X[LABELS[tag]].append(' '.join(i[1:1501]))
    return X


def cmp_my(x):
    return x[1]


class Securitas():
    def __init__(self, X, labels, voca_size, n_topic):
        self.LABELS = labels
        self.voca_size = voca_size
        self.n_topic = n_topic
        self.voca = self.get_vocaubulary(X, self.voca_size)
        self.lda = LatentDirichletAllocation(n_components=n_topic, doc_topic_prior=0.1, topic_word_prior=0.01)
        self.lda = self.fit(X)

    def fit(self, X):
        X = self.get_input_vectors(X)
        self.lda.fit(X)
        return self.lda


    def get_vocaubulary(self, X, need_size):
        vec =  CountVectorizer(min_df=1, ngram_range=(3,3),decode_error="ignore")
        X = vec.fit_transform(X)
        if need_size >= len(vec.get_feature_names()):
            need_size = len(vec.get_feature_names())

        # print('shape of X:', X.shape)
        X = X.toarray()
        X = np.sum(X, axis = 0)
        voca_indexs = {value:key for key, value in vec.vocabulary_.items()}
        X = sorted([(i,r) for i, r in enumerate(X)], key = cmp_my, reverse = True)
        res = {voca_indexs[item[0]]:i for i, item in enumerate(X[:need_size])}
        return res


    def get_input_vectors(self, X):
        vec = CountVectorizer(min_df = 1, ngram_range=(3,3), decode_error= "ignore", vocabulary= self.voca)
        X = vec.fit_transform(X)
        return X.toarray()

    def get_features(self, X):
        X_input_features = self.get_input_vectors(X)
        X_features = self.lda.transform(X_input_features)
        return X_features


def deal_to_binary(target, y):
    for i in range(len(y)):
        if y[i] != target:
            y[i] = 0
        else:
            y[i] = 1
    return y


def f1(p, r):
    return float(2*p*r) / float(p+r)


# LABELS = {'vimeo': 0, 'spotify': 1, 'voipbuster': 2, 'sinauc': 3, 'cloudmusic': 4, 'weibo': 5, 'baidu': 6, 'tudou': 7, 'amazon': 8, 'thunder': 9, 'gmail': 10, 'pplive': 11, 'qq': 12, 'taobao': 13, 'yahoomail': 14, 'itunes': 15, 'twitter': 16, 'jd': 17, 'sohu': 18, 'youtube': 19, 'youku': 20, 'netflix': 21, 'aimchat': 22, 'kugou': 23, 'skype': 24, 'facebook': 25, 'google': 26, 'mssql': 27, 'ms-exchange': 28}
# LABELS = {'audio': 0, 'browsing': 1, 'chat': 2, 'file': 3, 'mail': 4,
#           'p2p': 5, 'video': 6, 'voip': 7}
LABELS = {'reddit': 0, 'facebook': 1, 'NeteaseMusic': 2,
          'twitter': 3, 'qqmail': 4, 'instagram': 5,
          'weibo': 6, 'iqiyi': 7, 'imdb': 8,
          'TED': 9, 'douban': 10,
          'amazon': 11, 'youtube': 12, 'JD': 13,
          'youku': 14, 'baidu': 15,
          'google': 16, 'tieba': 17, 'taobao': 18,
          'bing': 19}

pp, rr, f1s = [[], [], []], [[], [], []], [[], [], []]


# filename is the same as BSNN
p_log('start construct_traffic')
X_total = construct_traffic('../bsnn/data/20_header_payload_all.traffic', LABELS)
for i, k in enumerate(X_total):
    p_log(i, ' ', len(k))


def go(X_total):
    securitas_time_logs = []
    for target in range(len(LABELS.keys())):
        p_log('Target: {}'.format(target))
        X1 = X_total[target]
        if len(X1) > 2000:
            X1 = list(np.random.choice(X1, size=[2000,]))
        len_negative = 2000 / (len(LABELS) - 1)
        len_negative = int(len_negative)
        p_log('len_negative: {}'.format(len_negative))
        X2 = []
        for i in range(len(X_total)):
            if i != target:
                X2 += list(np.random.choice(X_total[i], size=[len_negative,]))

        y1 = [1]*len(X1)
        y2 = [0]*len(X2)
        X = X1 + X2

        y = y1 + y2
        p_log('positve samples : {}, total: {}'.format(len(X1), len(X)))
        p_log('dataset ok')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 1)
        # y_train = deal_to_binary(14, y_train)
        # y_test = deal_to_binary(14, y_train)
        securitas_start_time = time()

        securitas = Securitas(X_train, LABELS, voca_size = 1500, n_topic = 45)

        securitas_train_time = time() - securitas_start_time

        p_log('securitas create ok')

        X_train_features = securitas.get_features(X_train)

        securitas_start_time = time()
        X_test_features = securitas.get_features(X_test)

        securitas_preprocess_time = time() - securitas_start_time

        p_log('securitas features ok, begin to train ML model')

        models = [DecisionTreeClassifier(), SVC(), MultinomialNB()]

        ML_time = {}
        for i in range(len(models)):
            model = models[i]
            p_log('model {}'.format(model.__class__.__name__))
            s_t = time()
            model.fit(X_train_features, y_train)
            time_fit = time() - s_t
            s_t = time()
            predicts = model.predict(X_test_features)
            time_pred = time() - s_t
            cmatrix = confusion_matrix(y_test, predicts)
            p_log(cmatrix)
            # p_sum = cmatrix.sum(axis = 1)
            # r_sum = cmatrix.sum(axis = 0)
            # p = cmatrix[1][1] / float(p_sum[1]+0.0001) + 0.0001
            # r = cmatrix[1][1] / float(r_sum[1]+0.0001) + 0.0001
            p, r, f1, _ = precision_recall_fscore_support(
                y_test, predicts, labels=[1,])
            pp[i].append(p)
            rr[i].append(r)
            f1s[i].append(f1)
            # f1_ = f1(p, r)
            p_log('precision: {}, recall: {}, f1: {}'.format(
                p, r, f1))
            p_log('time fit: {}, time predict: {}'.format(
                time_fit, time_pred))
            ML_time[model.__class__.__name__] = {
                'train': time_fit, 'test': time_pred}
        securitas_time_logs.append({
            'train': securitas_train_time,
            'preprocessing': securitas_preprocess_time,
            'mode': ML_time
        })
    p_log('Securitas time log: {}'.format(securitas_time_logs))


p_log('start train')
go(X_total)


names = [i.__class__.__name__ for i in [DecisionTreeClassifier(), SVC(), MultinomialNB()]]
data = {}
for i, n in enumerate(names):
    data[n+'_precision}'] = pp[i]
    data[n+'_recall'] = rr[i]
    data[n+'_f1score'] = f1s[i]
df = pd.DataFrame(data)
df.to_excel('securitas_results_dataset_20_new.xlsx')

p_log('ok')
