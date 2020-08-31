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

    def __init__(self, size, in_channels=1):
        super().__init__()
        self.width, self.height = size

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
        self.cnn = nn.Sequential(*self.layers)

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(in_channels, 10, kernel_size=(3, 21)),
        #     nn.ReLU(),
        # )
        #
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(10, 10, kernel_size=(3, 21)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # )
        #
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(10, 10, kernel_size=(4, 21)),
        #     nn.ReLU(),
        # )
        #
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(10, 10, kernel_size=(4, 21)),
        #     nn.ReLU(),
        # )

        # self.fc = nn.Linear(2540, 50)
        self.fc = nn.Linear(15930, 50)

    def init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 50)

    def forward(self, x):
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
        torch.nn.init.xavier_uniform_(self.weight, 10)
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
        batch_size = X.size(1)
        X = X.transpose(0, 1)  # (B, T, N)
        alignment_scores = X.matmul(self.weight.t())
        # print('AS: ', alignment_scores)
        # print('wT: ', self.weight.t().shape)
        # print('AS: ', alignment_scores.shape)
        # alpha [T x 1]
        attn_weights = nn.functional.softmax(alignment_scores, dim=0)
        # print('X: ', X.shape)
        # print('ATT: ', attn_weights.shape)
        # print('X: ', X.shape)
        # h_att [1 x N]
        return torch.bmm(attn_weights.transpose(1, 2), X)