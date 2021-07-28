import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
from utils import p_log
from time import time


class FocalLoss(nn.Module):
    def __init__(self, class_num, device, alpha=None, gamma=2,
                 size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.device = device
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)

        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda(self.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def evaluate_loss_acc(model, dataloder, alpha, gamma, NUM_CLASS, DEVICE,
                      test=False, flow=False, aggregate='sum_max'):
    loss_func = FocalLoss(NUM_CLASS, DEVICE, alpha, gamma, True)
    total_loss = 0
    y_hat, y = [], []

    time_logs = []

    for i in range(len(dataloder)):
        batch_X, batch_y = dataloder[i]

        batch_X = batch_X.cuda(DEVICE)
        batch_y = batch_y.cuda(DEVICE)

        s_t = time()
        model.eval()
        out1 = model(batch_X)
        time_logs.append(time() - s_t)

        # p_log('DEBUG: out1 in batch {}: {} (shape: {})'.format(out1, i, out1.shape))
        # the loss for flow classification when test is different to the one when train
        loss = loss_func(out1, batch_y)
        total_loss += loss.item()
        # ----------------
        # torch favor of argmax...
        if not test or not flow:
            y_hat += out1.max(1)[1].tolist()
            y += batch_y.tolist()
        else:
            assert len(set(batch_y.tolist())) == 1, 'one batch stands for '
            'one flow when test! unexpected batch_y: {}'.format(batch_y)
            if aggregate == 'count_max':
                # aggregate stragety: count_max
                cnt = Counter(out1.max(1)[1].tolist())
                cnt = [[v, k] for k, v in cnt]
                cnt.sort(key=lambda x: x[0], reverse=True)
                y_hat += [int(cnt[0][1]), ]
            else:
                # another aggregate strategy: sum_max
                y_hat += [int(torch.sum(out1, 0).max(0)[1].tolist()), ]
            y += [int(batch_y[0].tolist()), ]

    total_loss = total_loss / len(dataloder)
    y = np.array(y)
    y_hat = np.array(y_hat)
    # p_log('DEBUG: y_hat: {} (shape: {})'.format(y_hat, y_hat.shape))
    # p_log('DEBUG: y: {} (shape: {})'.format(y, y.shape))
    accuracy = accuracy_score(y, y_hat)
    p_log('DEBUG inference time per batch:',
          str(sum(time_logs) / len(time_logs)))

    return total_loss, accuracy, [y, y_hat]
