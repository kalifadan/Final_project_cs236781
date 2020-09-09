import torch
from model.blocks import ConvNet, SoftmaxAttention
from model.tcn import TemporalConvNet
from torch import nn


class TCNNet(nn.Module):
    def __init__(self, seq_len=20):
        super().__init__()
        self.seq_len = seq_len

        lst = []
        for i in range(seq_len):
            conv = ConvNet()
            lst.append(conv)
            self.add_module('conv{}'.format(i), conv)

        self.cnn_layers = lst

        output_size = 30  # TODO Figure this out
        self.tcn = TemporalConvNet(seq_len, 1)
        self.attention = SoftmaxAttention(seq_len * output_size)
        self.fc = nn.Linear(output_size, 2)

    def forward(self, X):
        """
        Perform forward propagation on a batch of data
        :param X: A tensor of shape (B, N, H, W) where (W, H) are the image dimensions,
        N is the number of images per sample (sequence length), and B is the batch size.
        """
        out = [self.cnn_layers[i](X[:, i, :].float().unsqueeze(1)) for i in range(self.seq_len)]
        out = torch.stack(out)
        out = self.tcn(out)
        out = self.attention(out)
        out = out.squeeze(1)
        out = self.fc(out)
        return out
