import random
from time import time
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import torch
import os
from utils import p_log
import h5py
from mpi4py import MPI


def pack(X, segment_len=8):    # X: b * l
    batch = X.size(0)
    length = X.size(1)
    l = int(length / segment_len)
    packets = []
    for i in range(batch):
        packet = []
        for j in range(l):
            packet.append(X[i][segment_len*j: segment_len*(j+1)].tolist())
        packets.append(packet)
    return torch.tensor(packets, dtype=torch.long)


def calculate_alpha(counter, mode='normal'):
    if mode == 'normal':
        alpha = torch.tensor(counter, dtype=torch.float32)
        alpha = alpha / alpha.sum(0).expand_as(alpha)
    elif mode == 'invert':
        alpha = torch.tensor(counter, dtype=torch.float32)
        alpha_sum = alpha.sum(0)
        alpha_sum_expand = alpha_sum.expand_as(alpha)
        alpha = (alpha_sum - alpha) / alpha_sum_expand
    # fill all zeros to ones
    alpha[alpha==0.] = 1.
    return alpha


class H5Iter:
    def __init__(self, st):
        self.corpus = []
        self.st = st
        
    def __call__(self, name, h5obj):
        if hasattr(h5obj, 'dtype') and name.endswith('/X'):
            self.corpus.append([name.split('/')[0], h5obj[:]])
            if len(self.corpus) % 200000 == 0:
                p_log(f'{time() - self.st:.4f}s with {len(self.corpus)} items.')


class Loader():
    def __init__(self, X_idx, corpus, batch_size, labels, segment_len=8,
                 shuffle=True, load_from_hdf5=False):
        self._debug = False
        if self._debug:
            debug_st = time()
        self.shuffle = shuffle
        self.X_idx = X_idx
        self.corpus = corpus
        self.labels = labels
        self.load_from_hdf5 = load_from_hdf5
        if self.shuffle:
            random.shuffle(self.X_idx)
        self.ys = {}
        if load_from_hdf5:
            for idx in self.X_idx:
                self.ys[idx] = labels[corpus[idx][0]]
        else:
            for idx in self.X_idx:
                self.ys[idx] = labels[
                    corpus[idx].split()[0].split('//')[0]]
        self.alpha = Counter(list(self.ys.values()))
        for i in labels.values():
            if i not in self.alpha:
                self.alpha[i] = 0
        # TODO(DCMMC): consistent with FlowLoader?
        self.alpha = calculate_alpha(
            [self.alpha[k] for k in sorted(self.alpha.keys())],
            mode='invert')
        self.segment_len = segment_len
        self.batch_size = batch_size
        self.num_samples = len(X_idx)
        self.corpus = corpus
        if self._debug:
            p_log('finish __init__ for class Loader after {}s.'.format(
                time() - debug_st
            ))
            p_log('Alpha: {}\n'.format(self.alpha))

    def __len__(self):
        return int(np.ceil(len(self.X_idx) / self.batch_size))

    def __getitem__(self, idx):
        if self._debug:
            st = time()
        batch_len = []
        batch_X, batch_y = [], []
        if self.load_from_hdf5:
            for pkt_idx in self.X_idx[idx * self.batch_size: (idx+1) * self.batch_size]:
                batch_X.append(self.corpus[pkt_idx][1])
                batch_len.append(len(batch_X[-1]))
                batch_y.append(self.ys[pkt_idx])
        else:
            for pkt_idx in self.X_idx[idx * self.batch_size: (idx+1) * self.batch_size]:
                batch_X.append([int(b, 16) for b in self.corpus[pkt_idx].split()[1:]])
                batch_len.append(len(batch_X[-1]))
                batch_y.append(self.ys[pkt_idx])
        # maximum length of packet maybe 65535bytes, which is extremely large!
        # Therefore we truncate large packets to 1500bytes.
        maxlen = min(1500, max(batch_len))
        batch_X = [np.append(x, [256] * (maxlen - len(x))) if (
                maxlen > len(x)) else x[:maxlen] for x in batch_X]
        batch_X = torch.tensor(batch_X, dtype=torch.long)
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        batch_X = pack(batch_X, segment_len=self.segment_len)
        if idx == len(self) - 1 and self.shuffle:
            p_log('shuffle dataloader')
            random.shuffle(self.X_idx)
        if self._debug:
            p_log('getitem {}, shape: {}, with {}s.\n'.format(
                idx, batch_X.shape, time() - st))
        return (batch_X, batch_y)


def _get_dataloader_packet(filename, labels, test_percent, batch_size,
                           segment_len=8, shuffle=True):
    # Turn file to X and y. percent is test_size
    s_t = time()
    if filename.endswith('.hdf5'):
        p_log('Load from hdf5 file.')
        load_from_hdf5 = True
        corpus = []
        f_h5 = h5py.File(os.path.join('data', filename), 'r',
                         driver='mpio', comm=MPI.COMM_WORLD)
        h5iter = H5Iter(s_t)
        f_h5.visititems(h5iter)
        corpus = h5iter.corpus
        f_h5.close()
    else:
        with open(os.path.join('data', filename), 'r', encoding='utf-8',
                  errors='ignore') as f:
            corpus = f.readlines()
    p_log('open dataset and load it into corpus, done with {}s\n'.format(
        time() - s_t))
    X_idx = list(range(len(corpus)))
    if test_percent < 1.0:
        X_idx_train, X_idx_test, _, _ = train_test_split(
            X_idx, [0 for _ in X_idx], test_size=test_percent, random_state=0
        )
    else:
        X_idx_train, X_idx_test = [], X_idx
    p_log('test_percent is {}, len(X_train)={}, len(X_test)={}\n'.format(
        test_percent, len(X_idx_train), len(X_idx_test)))
    if test_percent < 1.0:
        train_loader = Loader(
            X_idx_train, corpus, batch_size, labels,
            segment_len=segment_len, shuffle=shuffle,
            load_from_hdf5=load_from_hdf5
        )
    else:
        train_loader = None
    test_loader = Loader(
        X_idx_test, corpus, batch_size, labels,
        segment_len=segment_len, shuffle=shuffle,
        load_from_hdf5=load_from_hdf5
    )
    p_log('split dataset done with {}s\n'.format(time() - s_t))
    return train_loader, test_loader


################################################
# DCMMC: new getloader for flow classification #
################################################


class FlowLoader():
    # if construct test dataset, we can specify how many
    # packets we will use to test for each flow
    # X is the paths of flows
    # we dont use y as all, instead the ys are stored in hdf5 file
    # labels: a dict with key=applicatin name, value=numerical label
    def __init__(self, X, labels, batch_size, filename, alpha, test_dataset=False,
                 first_k_packets=3, shuffle=True, segment_len=8, buffer_size=2048):
        self._debug = False
        if self._debug:
            debug_s_t = time()
        self.test_dataset = test_dataset
        self.alpha = alpha
        self.shuffle = shuffle
        self.f_h5 = h5py.File(filename, 'r')
        self.segment_len = segment_len
        self.buffer = []
        self.buffer_offset = 0
        self.buffer_size = buffer_size
        # the data only store the hdf5 paths for all samples
        self.data = []
        for flow in X:
            keys = list(self.f_h5[flow].keys())
            # one batch stands for one flow when test
            if self.test_dataset:
                keys = sorted([int(k) for k in keys])
                batch_size = first_k_packets
                keys = keys[:first_k_packets]
                self.data.append([['/'.join([flow, str(p), 'X']),
                               '/'.join([flow, str(p), 'y'])] for p in keys])
            else:
                self.data += [['/'.join([flow, str(p), 'X']),
                               '/'.join([flow, str(p), 'y'])] for p in keys]
        if self._debug:
            p_log('finish get data after {}s.'.format(time() - debug_s_t))
        if shuffle:
            p_log('shuffle dataloader')
            random.shuffle(self.data)
        if not self.test_dataset:
            self.num_samples = len(self.data)
            self.num_batch = int(np.ceil(self.num_samples / batch_size))
        else:
            self.num_samples = len(self.data) * first_k_packets
            self.num_batch = len(self.data)
        self.batch_size = batch_size
        p_log('DataLoader for {} done. batch_size: {}\n'.format(
            'Test' if self.test_dataset else 'Train', batch_size))
        p_log('Alpha: {}\n'.format(self.alpha))
        if self._debug:
            p_log('dataloader constructured after {}s.\n'.format(time() - debug_s_t))

    def __len__(self):
        return self.num_batch

    def __getitem__(self, idx):
        if self._debug:
            debug_s_t = time()
        if idx < 0 or idx >= len(self):
            raise IndexError(f'Index {idx} out of range [0, {len(self)})')
        if len(self.buffer) + self.buffer_offset == idx:
            self.buffer_offset += len(self.buffer)
            self.buffer = []
            for idx_buff in range(idx, min(len(self), idx + self.buffer_size)):
                idx_loc = slice(
                    idx_buff * self.batch_size,
                    (idx_buff + 1) * self.batch_size
                ) if not self.test_dataset else idx_buff
                batch_X = []
                batch_y = []
                batch_len = []
                for b_path in self.data[idx_loc]:
                    sample_x = self.f_h5[b_path[0]][:]
                    sample_y = int(self.f_h5[b_path[1]][...])
                    batch_len.append(len(sample_x))
                    batch_X.append(sample_x)
                    batch_y.append(sample_y)
                if self.test_dataset:
                    assert len(set(batch_y)) == 1, 'one batch stands for one flow when test!'
                    ' unexpected batch_y: {}, batch_idx: {}'.format(batch_y, idx)
                # maximum length of packet maybe 65535bytes, which is extremely large!
                # Therefore we truncate large packets to 1500bytes.
                max_len = min(1500, max(batch_len))
                # padding all sample with 256 to ensure their lengths are maxlen
                batch_X = [np.append(x, [256] * (max_len - len(x))) if (
                    max_len > len(x)) else x[:max_len] for x in batch_X]
                self.buffer.append([batch_X, batch_y])
        batch_X, batch_y = self.buffer[idx - self.buffer_offset]
        batch_X = torch.tensor(batch_X, dtype=torch.long)
        batch_X = pack(batch_X, segment_len=self.segment_len)
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        if idx == len(self) - 1:
            if self.shuffle:
                p_log('shuffle dataloader')
                random.shuffle(self.data)
            self.buffer = []
            self.buffer_offset = 0
        if self._debug:
            p_log('getitem {}, shape: {}, with {}s.\n'.format(
                idx, batch_X.shape, time() - debug_s_t))
        return (batch_X, batch_y)


def _get_dataloader_flow(filename, labels, test_percent, batch_size,
                         first_k_packets, segment_len=8, shuffle=True):
    # transform traffic file to h5 file
    s_t = time()
    xs_path = []
    ys = []
    with open(os.path.join('data', filename), 'r', encoding='utf-8',
              errors='ignore') as f:
        byte_to_ix = {hex(i)[2:].upper().zfill(2): i for i in range(256)}
        p_log('start creating hdf5 dataset')
        f_h5 = os.path.join('data', filename.split('.')[0] + '.hdf5')
        if os.path.isfile(f_h5):
            p_log('hdf5 file already exists, skip creating.')
            f_h5 = None
            f_h5 = h5py.File(filename, 'r')
        else:
            f_h5 = h5py.File(f_h5, 'w')
            packet = f.readline()
            while packet:
                data_bytes = packet.split()
                label, flow_idx, packet_idx = data_bytes[0].split('//')
                label_numerical = labels[label]
                flow_path = '/'.join([label, flow_idx])
                data_bytes = data_bytes[1:]
                data_bytes_numerical = [byte_to_ix[b] for b in data_bytes]
                f_h5.create_dataset('/'.join([flow_path, packet_idx, 'X']),
                                    data=np.array(data_bytes_numerical,
                                                  dtype=np.uint8))
                f_h5['/'.join([flow_path, packet_idx, 'y'])] = int(
                    label_numerical)
                if flow_path not in xs_path:
                    xs_path.append(flow_path)
                    ys.append([label_numerical, 1])
                else:
                    ys[-1][1] += 1
                packet = f.readline()
                f_h5.close()
        p_log(
            'Done with {}s, there are {} flows in total\n'.format(
                time() - s_t, len(xs_path)))
    s_t = time()
    p_log('start constructing DataLoader.\n')
    if test_percent == 1.0:
        X_train, y_train = [], []
        tmp = list(zip(xs_path, ys))
        random.shuffle(tmp)
        X_test, y_test = zip(*tmp)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            xs_path, ys, test_size=test_percent, random_state=0)
    alpha_train  = {i: 0 for i in range(len(labels))}
    alpha_test = {i: 0 for i in range(len(labels))}
    for k, v in y_train:
        alpha_train[k] += v
    for k, v in y_test:
        alpha_test[k] += v
    if test_percent < 1.0:
        alpha_train = calculate_alpha(
            [alpha_train[k] for k in sorted(alpha_train.keys())],
            mode='invert'
        )
        train_loader = FlowLoader(
            X_train, labels, batch_size,
            os.path.join('data', filename.split('.')[0] + '.hdf5'),
            alpha_train, shuffle=shuffle,
            first_k_packets=first_k_packets, segment_len=segment_len)
    else:
        train_loader = None
    alpha_test = calculate_alpha(
        [alpha_test[k] for k in sorted(alpha_test.keys())],
        mode='invert'
    )
    test_loader = FlowLoader(
        X_test, labels, batch_size,
        os.path.join('data', filename.split('.')[0] + '.hdf5'),
        alpha_test, shuffle=shuffle,
        test_dataset=True, first_k_packets=first_k_packets,
        segment_len=segment_len)
    p_log('split dataset done with {}s\n'.format(time() - s_t))
    return train_loader, test_loader


# Turn file to X and y. percent is test_size
def get_dataloader(filename, labels, test_percent, batch_size, flow=False,
                   first_k_packets=None, segment_len=8, shuffle=True):
    if flow:
        return _get_dataloader_flow(filename, labels, test_percent,
                                    batch_size, first_k_packets,
                                    segment_len=segment_len, shuffle=shuffle)
    else:
        return _get_dataloader_packet(filename, labels, test_percent,
                                      batch_size, segment_len=segment_len,
                                      shuffle=shuffle)
