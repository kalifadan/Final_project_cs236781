import torch
from torch import nn


class ConvNet(nn.Module):

    @staticmethod
    def _conv_net2d(in_channels, kernel_size=(3, 21)):
        # return [
        #     nn.Conv2d(in_channels, 10, kernel_size=kernel_size),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(10),
        #
        #     nn.Conv2d(10, 10, kernel_size=kernel_size),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #     nn.BatchNorm2d(10),
        #
        #     nn.Conv2d(10, 10, kernel_size=(4, 21)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(10),
        #
        #     nn.Conv2d(10, 10, kernel_size=(4, 21)),
        #     nn.ReLU(),
        # ]
        return [
            nn.Conv2d(in_channels, 8, kernel_size=(32, 64)),
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 8, kernel_size=(32, 64)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # nn.BatchNorm2d(10),
            #
            # nn.Conv2d(10, 10, kernel_size=(4, 21)),
            # nn.ReLU(),
            # nn.BatchNorm2d(10),
            #
            # nn.Conv2d(10, 10, kernel_size=(4, 21)),
            # nn.ReLU(),
        ]


    @staticmethod
    def _conv_net1d(in_channels, stride, padding, dilation):
        return [
            nn.Conv1d(in_channels, 4, 100, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(4, 8, 100, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(8, 16, 100, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ]

    def __init__(self, in_channels=1, batch=True, output_size=50):
        super().__init__()
        self.batch = batch
        self.layers = ConvNet._conv_net2d(in_channels)
        self.cnn = nn.Sequential(*self.layers)
        self.fc = nn.Linear(14880, output_size)

    def forward(self, x):
        x = x.float()
        if not self.batch:
            x = x.unsqueeze(1)

        out = self.cnn(x)
        out = out.flatten(start_dim=1)
        out = self.fc(out)
        return out


class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_gates):
        super().__init__()
        self.bi_rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_gates,
                                   batch_first=False, bidirectional=True)

    def forward(self, X):
        """
        Perform forward propagation on a batch of data
        :param X: A tensor of shape (L, B, H) where L is the sequence length, H is the number of features
        per sequence unit and B is the batch size
        :return: A tensor of shape (L, B, 2 * H) containing the outputs of the BRNN
        """
        output, hn = self.bi_rnn(X)
        return output


attention_desc = r"""
Notations:
* $Y = \left[ y_1, \ldots, y_T \right]$ – the input matrix of size $\left( N \times T \right)$, where $N$ is the number of features in a single output vector of the BRNN
* $w_\mathrm{att}$ – The parameters of the attention model, of size $\left( N \times 1 \right)$, where $N$ is the number of features in a single output vector of the BRNN
* $\alpha$ – The attention weights, given as $\alpha = \mathrm{softmax} \left( w_\mathrm{att}^T Y \right)$. This is an element-wise softmax, where the output size of $\alpha$ is $\left( 1 \times T \right)$
* $h_\mathrm{att}$ – Output of the attention mechanism, given by $h_\mathrm{att} = Y \alpha^T$, of size $\left( N \times 1 \right)$, i.e. a vector of $N$ features.
"""


class SoftmaxAttention(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the attention model
        :param input_size: Number of inputs to accept. during forward propagation
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, X):
        """
        Perform forward propagation
        :param X: A tensor of shape (T, B, N) where T is the sequence len, B is the batch size and N is the number of
        features in each sequence node
        :return: A tensor of shape (B, 1, N) representing the weighted attention features for each sample in the batch
        """
        # X [T, B, N]
        # weight [N x 1]

        X = X.transpose(0, 1)  # [B, T, N]
        # weight is [1, N]
        alignment_scores = X.matmul(self.weight.t())  # [B, T, 1]
        alignment_scores = alignment_scores.squeeze(-1)  # [B, T]

        # alpha [B, T]
        attn_weights = nn.functional.softmax(alignment_scores, dim=1)
        # h_att [B, 1, T] bmm [B, T, N] -> [B, 1, N]
        return torch.bmm(attn_weights.unsqueeze(-2), X)
