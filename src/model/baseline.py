import torch
from model.blocks import ConvNet, BRNN, SoftmaxAttention
from torch import nn


class Baseline(nn.Module):
    def __init__(self, seq_len=20, add_brnn=True, hidden_size=100):
        super().__init__()
        self.seq_len = seq_len

        lst = []
        for i in range(seq_len):
            conv = ConvNet()
            lst.append(conv)
            self.add_module('conv{}'.format(i), conv)

        self.cnn_layers = lst

        self.support_brnn = add_brnn
        if add_brnn:
            self.brnn = BRNN(50, hidden_size, seq_len)

        output_coef = 1 + int(add_brnn)
        self.attention = SoftmaxAttention(output_coef * hidden_size)
        self.fc = nn.Linear(output_coef * hidden_size, 2)
        self.brnn_fc = nn.Linear(output_coef * hidden_size * seq_len, 2)

    def forward(self, X):
        """
        Perform forward propagation on a batch of data
        :param X: A tensor of shape (B, N, H, W) where (W, H) are the image dimensions,
        N is the number of images per sample (sequence length), and B is the batch size.
        """
        out = [self.cnn_layers[i](X[:, i, :].float().unsqueeze(1)) for i in range(self.seq_len)]
        out = torch.stack(out)
        out = self.brnn(out)
        out = self.attention(out)
        out = out.squeeze(1)
        out = self.fc(out)
        return out
