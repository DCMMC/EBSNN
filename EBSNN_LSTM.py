import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import p_log


class EBSNN_LSTM(nn.Module):
    def __init__(self, num_class, embedding_dim, device,
                 bidirectional=True, segment_len=8, dropout_rate=0.5):
        super(EBSNN_LSTM, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda:{}".format(device))
        self.segment_len = segment_len
        self.rnn_dim = 100
        # if bi-direction
        self.rnn_directions = 2 if bidirectional else 1

        # 256 is 'gg', will be set [0,0..0]
        self.byte_embed = nn.Embedding(257, self.embedding_dim, padding_idx=256)
        # to one-hot
        self.byte_embed.requires_grad = True

        self.rnn1 = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.rnn_dim,
                            batch_first=True, dropout=dropout_rate,
                            bidirectional=(self.rnn_directions == 2))
        self.rnn2 = nn.LSTM(self.rnn_directions * self.rnn_dim, self.rnn_dim,
                            batch_first=True, dropout=dropout_rate,
                            bidirectional=(self.rnn_directions == 2))

        self.hc1 = torch.randn(self.rnn_dim, 1)
        self.hc2 = torch.randn(self.rnn_dim, 1)

        self.fc1 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc3 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.num_class)

        self.dropout = nn.Dropout(0.5)

    '''
    x: b * l * 8
    lengths: (b,), every batch's length
    '''

    def forward(self, x):
        batch_size = x.size(0)
        x = self.byte_embed(x)  # b * l * 8 * 257

        out1, (h_n, c_n) = self.rnn1(x.view(-1, self.segment_len, self.embedding_dim))
        # bl * 8 * 100
        h = torch.tanh(self.fc1(out1.contiguous().view(
            -1, self.rnn_directions * self.rnn_dim))).view(
                out1.size(0), out1.size(1), -1)
        h = h.cuda(self.device)
        self.hc1 = self.hc1.cuda(self.device)
        weights = (torch.matmul(h, self.hc1)).view(-1, self.segment_len)
        weights = F.softmax(weights, dim=1)
        weights = weights.view(-1, 1, self.segment_len)
        # b * l * 200
        out2 = torch.matmul(weights, out1).view(
            batch_size, -1, self.rnn_dim * self.rnn_directions)

        out3, (h1_n, h2_n) = self.rnn2(out2)  # out3: b * l * 200
        h2 = torch.tanh(self.fc2(out3.contiguous().view(
            -1, self.rnn_dim * self.rnn_directions))).view(
                out3.size(0), out3.size(1), -1)
        h2 = h2.cuda(self.device)
        self.hc2 = self.hc2.cuda(self.device)
        weights2 = F.softmax((torch.matmul(h2, self.hc2)).view(
            batch_size, -1), dim=1).view(batch_size, 1, -1)
        out4 = torch.matmul(weights2, out3).view(
            batch_size, self.rnn_dim * self.rnn_directions)

        out = self.dropout(self.fc3(out4))
        return out
