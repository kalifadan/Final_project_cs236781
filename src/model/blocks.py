import torch
from torch import nn


class ConvNet(nn.Module):
    # @staticmethod
    # def _calculate_layers_fc_size(layers, width):
    #     size = -1
    #     for layer in layers:
    #         if isinstance(layer, nn.MaxPool2d):
    #             size /= 2  # TODO Change
    #         elif isinstance(layer, nn.Conv2d):
    #             if size == -1:
    #                 size =

    # TODO Remove fc_size
    def __init__(self, size, in_channels=1, batch=True, input_size=7500, output_size=50):
        super().__init__()
        self.width, self.height = size
        self.batch = batch
        self.layers = [
            nn.Conv2d(in_channels, 10, kernel_size=(3, 21)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # nn.Conv2d(10, 10, kernel_size=(3, 21)),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # nn.Conv2d(10, 10, kernel_size=(4, 21)),
            # nn.ReLU(),
            #
            # nn.Conv2d(10, 10, kernel_size=(4, 21)),
            # nn.ReLU(),
        ]

        kernel_size = 5
        stride = 1
        padding = 0
        dilation = 1
        self.layers = [
            nn.Conv1d(in_channels, 4, 5, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(4, 8, 5, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(8, 16, 5, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ]

        # self.layers = [
        #     nn.Conv2d(in_channels, 32, kernel_size=(3, 21)),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(32, 10, kernel_size=(3, 21)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #
        #     # nn.Conv2d(10, 10, kernel_size=(4, 21)),
        #     # nn.ReLU(),
        #     #
        #     # nn.Conv2d(10, 10, kernel_size=(4, 21)),
        #     # nn.ReLU(),
        # ]

        self.cnn = nn.Sequential(*self.layers)
        fc_size = int(((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
        # self.fc = nn.Linear(2540, 50)
        self.fc = nn.Linear(59904, output_size)

    def forward(self, x):
        x = x.float()
        if not self.batch:
            x = x.unsqueeze(1)
        out = self.cnn(x)
        # print('After conv: ', out)
        # print(out.shape)
        out = out.flatten(start_dim=1)
        out = self.fc(out)
        # print('After conv-fc: ', out)
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


class SoftmaxAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)
        # print(self.weight.data)

    def forward(self, X):
        """
        Perform forward propagation
        :param X: A tensor of shape (T, B, N) where T is the sequence len, B is the batch size and N is the number of
        features in each sequence node
        :return: A tensor of shape (B, 1, N) representing the weighted attention features for each sample in the batch
        """
        # print('ATT:', X.shape)
        # X [T, B, N]
        # weight [N x 1]

        X = X.transpose(0, 1)  # (B, T, N)
        alignment_scores = X.matmul(self.weight.t())

        # alpha [T x 1]
        attn_weights = nn.functional.softmax(alignment_scores, dim=1)

        # h_att [1 x N]
        return torch.bmm(attn_weights.transpose(1, 2), X)
