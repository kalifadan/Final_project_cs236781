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
            torch.nn.init.xavier_uniform_(m.weight, gain=5)
            m.bias.data.normal_(0, 1)
        elif isinstance(m, ConvNet):
            m.init_weights()

    def forward(self, X):
        """
        Perform forward propagation on a batch of data
        :param X: A tensor of shape (B, N, H, W) where (W, H) are the image dimensions,
        N is the number of images per sample, and B is the batch size
        """
        out = [self.cnn_layers[i](X[:, i, :].float().unsqueeze(1)) for i in range(self.seq_len)]
        out = torch.stack(out)

        # print('Before BRNN: ', out.shape)
        out = self.brnn(out)

        out = self.attention(out)
        out = out.squeeze(1)

        # out = self.fc(out.transpose(0, 1).flatten(start_dim=1))
        out = self.fc(out)
        out = out.squeeze(1)
        # print('After FC: ', out[:10])
        return F.softmax(out, dim=1)
