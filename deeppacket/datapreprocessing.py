from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import torch
import random
from time import time


class Loader():
    def __init__(self, X_idx, corpus, batch_size, labels,
                 shuffle=True):
        self._debug = False
        if self._debug:
            debug_st = time()
        self.shuffle = shuffle
        self.X_idx = X_idx
        self.corpus = corpus
        self.labels = labels
        if self.shuffle:
            random.shuffle(self.X_idx)
        self.ys = {}
        for idx in self.X_idx:
            self.ys[idx] = labels[
                corpus[idx].split()[0].split('//')[0]]
        self.alpha = Counter(list(self.ys.values()))
        for i in labels.values():
            if i not in self.alpha:
                self.alpha[i] = 0
        self.batch_size = batch_size
        self.num_samples = len(X_idx)
        self.corpus = corpus
        if self._debug:
            print('finish __init__ for class Loader after {}s.'.format(
                time() - debug_st
            ), flush=True)

    def __len__(self):
        return int(np.ceil(len(self.X_idx) / self.batch_size))

    def __getitem__(self, idx):
        if self._debug:
            st = time()
        batch_len = []
        batch_X, batch_y = [], []
        for pkt_idx in self.X_idx[idx * self.batch_size: (idx+1) * self.batch_size]:
            batch_X.append([int(b, 16) for b in self.corpus[pkt_idx].split()[1:]])
            batch_len.append(len(batch_X[-1]))
            batch_y.append(self.ys[pkt_idx])
        # maxlen = min(1500, max(batch_len))
        # DeepPacket keep or zero-pad every packet to fixed length: 1500
        maxlen = 1500
        batch_X = [np.append(x, [0] * (maxlen - len(x))) if (
                maxlen > len(x)) else x[:maxlen] for x in batch_X]
        batch_X = torch.tensor(batch_X, dtype=torch.float32)
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        # batch_X = pack(batch_X, segment_len=self.segment_len)
        if idx == len(self) - 1 and self.shuffle:
            print('shuffle dataloader', flush=True)
            random.shuffle(self.X_idx)
            del self.ys
            self.ys = {}
            for idx in self.X_idx:
                self.ys[idx] = self.labels[
                    self.corpus[idx].split()[0].split('//')[0]]
        if self._debug:
            print('getitem {}, shape: {}, with {}s.\n'.format(
                idx, batch_X.shape, time() - st), flush=True)
        return (batch_X, batch_y)


def get_dataloader(filename, labels,
                   test_percent, batch_size,
                   shuffle=True):
    # Turn file to X and y. percent is test_size
    s_t = time()
    with open(filename, 'r', encoding='utf-8',
              errors='ignore') as f:
        corpus = f.readlines()
    print('open dataset and load it into corpus, done with {}s\n'.format(
        time() - s_t), flush=True)
    X_idx = list(range(len(corpus)))
    if test_percent < 1.0:
        X_idx_train, X_idx_test, _, _ = train_test_split(
            X_idx, [0 for _ in X_idx], test_size=test_percent, random_state=0
        )
    else:
        X_idx_train, X_idx_test = [], X_idx
    print('test_percent is {}, len(X_train)={}, len(X_test)={}\n'.format(
        test_percent, len(X_idx_train), len(X_idx_test)), flush=True)
    if test_percent < 1.0:
        train_loader = Loader(
            X_idx_train, corpus, batch_size, labels,
            shuffle=shuffle)
    else:
        train_loader = None
    test_loader = Loader(
        X_idx_test, corpus, batch_size, labels,
        shuffle=shuffle)
    print('split dataset done with {}s\n'.format(time() - s_t), flush=True)
    return train_loader, test_loader
