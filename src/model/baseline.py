import torch
from model.blocks import ConvNet, BRNN, SoftmaxAttention
from torch import nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, seq_len=20, input_size=(375, 20)):
        super().__init__()
        self.seq_len = seq_len

        lst = []
        for i in range(seq_len):
            conv = ConvNet(input_size)
            lst.append(conv)
            self.add_module('conv{}'.format(i), conv)

        self.cnn_layers = lst
        self.brnn = BRNN(50, 50, seq_len)
        self.attention = SoftmaxAttention(100)
        self.fc = nn.Linear(100, 2)
        self.apply(Baseline.init_weights)

    @staticmethod
    def init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, X):
        """
        Perform forward propagation on a batch of data
        :param X: A tensor of shape (B, N, H, W) where (W, H) are the image dimensions,
        N is the number of images per sample, and B is the batch size
        """
        # print('[Input]: ', (X[:, 0, :].float()[0][0] - X[:, 1, :].float()[0][0]).std())
        # print('input #1: ', X[:, 0, :].float()[0][0])
        # print('input #2: ', X[:, 1, :].float()[0][0])
        # print(X[0,0,:].float().unsqueeze(1), X[0,1,:].float().unsqueeze(1))
        X = X
        out = [self.cnn_layers[i](X[:, i, :].float().unsqueeze(1)) for i in range(self.seq_len)]
        out = torch.stack(out)
        # print('[After CNN]: ', (out[0][0] - out[0][1]).std())
        # print('Equal:', out[0][0] == out[0][1])
        # print('After CNN: ', out[0][0], out[0][1])

        # print('Before BRNN: ', out)
        out = self.brnn(out)
        # print('After BRNN: ', out.squeeze(1))
        out = self.attention(out.squeeze(1))
        out = out.squeeze(1)
        # print('After ATT: ', out)
        # print('ATT Weights:', dict(self.attention.named_parameters())['weight'])

        out = self.fc(out)
        out = out.squeeze(1)
        print('After FC: ', out[:10])
        return F.softmax(out, dim=1)
