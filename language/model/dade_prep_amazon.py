import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from layers import *


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.fc3 = nn.Linear(5000, 50)

        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        # if self.use_drop:
        #     self.drop1 = nn.Dropout()

        if self.use_bn:
            self.fc3_bn = nn.BatchNorm1d(50)

    def forward(self, x):
        # if self.use_drop:
        #     x = self.drop1(x)
        #
        if self.use_bn:
            x = torch.sigmoid(self.fc3_bn(self.fc3(x)))
        else:
            x = torch.sigmoid(self.fc3(x))
        return x


class Classifier(nn.Module):
    def __init__(self, args, n_classes=2):
        super(Classifier, self).__init__()
        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        if self.use_drop:
            self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(50, n_classes)

    def forward(self, x):
        if self.use_drop:
            x = self.drop1(x)
        x = self.fc4(x)
        return x



