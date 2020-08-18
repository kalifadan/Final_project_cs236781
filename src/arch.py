import torch
from torch import nn


class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_gates):
        super(BRNN, self).__init__()
        self.bi_grus = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_gates,
                                    batch_first=False, bidirectional=True)

    def forward(self, X):
        pass
        # |Y| = 100 x 20
        # |w_att| = 100 x 1
        # |alpha_transpose| = ([1 x 100] x [100 x 20])^T = 20 x 1