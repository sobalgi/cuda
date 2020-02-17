import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from layers import *


class Encoder(nn.Module):
    def __init__(self, args,
                 input_size=5000,
                 hidden_sizes=[1000, 500],
                 output_size=128,
                 dropout=0,
                 batch_norm=False):
        super(Encoder, self).__init__()
        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel
        self.hidden_sizes = hidden_sizes
        self.net = nn.Sequential()
        num_layers = len(hidden_sizes)
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(input_size, hidden_sizes[0]))
            else:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if batch_norm:
                self.net.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_sizes[i]))
            self.net.add_module('f-relu-{}'.format(i), nn.ReLU())

        if dropout > 0:
            self.net.add_module('f-dropout-final', nn.Dropout(p=dropout))
        self.net.add_module('f-linear-final', nn.Linear(hidden_sizes[-1], output_size))
        if batch_norm:
            self.net.add_module('f-bn-final', nn.BatchNorm1d(output_size))
        self.net.add_module('f-relu-final', nn.ReLU())

    def forward(self, input):
        return self.net(input)


class Classifier(nn.Module):
    def __init__(self, args,
                 num_layers=1,
                 input_size=128,
                 hidden_size=128,
                 output_size=2,
                 dropout=0.5,
                 batch_norm=False):
        super(Classifier, self).__init__()
        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        assert num_layers >= 0, 'Invalid clayer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        #for i in range(num_layers):
        #    if dropout > 0:
        #        self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
        #    if i == 0:
        #        self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
        #    else:
        #        self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
        #    if batch_norm:
        #        self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
        #    self.net.add_module('p-relu-{}'.format(i), nn.ReLU())
        if dropout > 0:
             self.net.add_module('p-dropout-{}'.format(0), nn.Dropout(p=dropout))

        self.net.add_module('p-linear-final', nn.Linear(input_size, output_size))
        # self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)  # return logits


