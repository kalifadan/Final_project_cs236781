import torch
from model.blocks import ConvNet, BRNN, SoftmaxAttention
from torch import nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, seq_len=20, input_size=(375, 20), add_brnn=True, hidden_size=100):
        super().__init__()
        self.seq_len = seq_len

        lst = []
        for i in range(seq_len):
            conv = ConvNet(input_size)
            lst.append(conv)
            self.add_module('conv{}'.format(i), conv)

        self.cnn_layers = lst

        self.support_brnn = add_brnn
        if add_brnn:
            self.brnn = BRNN(50, hidden_size, seq_len)

        output_coef = 1 + int(add_brnn)
        self.attention = SoftmaxAttention(output_coef * hidden_size)
        self.fc = nn.Linear(output_coef * hidden_size, 2)
        # self.apply(Baseline.init_weights)

    @staticmethod
    def init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.normal_(0, 1)
        # elif isinstance(m, ConvNet):
        #     m.init_weights()

    def forward(self, X):
        """
        Perform forward propagation on a batch of data
        :param X: A tensor of shape (B, N, H, W) where (W, H) are the image dimensions,
        N is the number of images per sample, and B is the batch size
        """
        # print('X=', X.shape)  # (B, N, H, W)
        # print('Single X=', X[:, 0, :].float().unsqueeze(1).shape)
        # return
        out = [self.cnn_layers[i](X[:, i, :].float().unsqueeze(1)) for i in range(self.seq_len)]
        print(out[0].shape)
        print(out[0])
        print(out[1])
        out = torch.stack(out)
        print('After CNN=', out.shape)
        out = self.brnn(out)
        print('After BRNN=', out.shape, '\n', out)
        out = self.attention(out)
        out = out.squeeze(1)
        out = self.fc(out)
        # out = self.fc(out.squeeze(0))
        return out
