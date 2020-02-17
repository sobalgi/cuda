import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nninit
from torch.distributions import Gumbel


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        # nninit.kaiming_normal_(self.conv1_1.weight, nonlinearity='relu')
        # nninit.kaiming_normal_(self.conv2_1.weight, nonlinearity='relu')
        # nninit.kaiming_normal_(self.conv2_2.weight, nonlinearity='relu')

        # self.drop1 = nn.Dropout()
        # self.fc3 = nn.Linear(1024, 256)
        # self.fc3_bn = nn.BatchNorm1d(256)
        # self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1024)

        # x = self.drop1(x)
        # x = F.relu(self.fc3_bn(self.fc3(x)))
        # x = self.fc4(x)
        return x


class Classifier(nn.Module):
    def __init__(self, args, n_classes=10):
        super(Classifier, self).__init__()

        # self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        # self.conv1_1_bn = nn.BatchNorm2d(32)
        # self.pool1 = nn.MaxPool2d((2, 2))
        #
        # self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        # self.conv2_1_bn = nn.BatchNorm2d(64)
        # self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        # self.conv2_2_bn = nn.BatchNorm2d(64)
        # self.pool2 = nn.MaxPool2d((2, 2))

        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, n_classes)

        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        if self.use_drop:
            self.drop1 = nn.Dropout()

        if self.use_bn:
            self.fc3_bn = nn.BatchNorm1d(256)

        if self.use_gumbel:
            self.gumbel_sampler = Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
            self.gumbel_temperature = nn.Linear(n_classes, 1)
            # nninit.xavier_normal_(self.gumbel_temperature.weight)

        # nninit.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        # nninit.xavier_normal_(self.fc4.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, x):
        # x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))
        #
        # x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        # x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        # x = x.view(-1, 1024)

        if self.use_drop:
            x = self.drop1(x)
        #
        if self.use_bn:
            x = F.relu(self.fc3_bn(self.fc3(x)))
        else:
            x = F.relu(self.fc3(x))
        x = self.fc4(x)

        if self.use_gumbel:
            t0 = 0.2
            if self.training:  # use gumbel softmax only for training
                # calculate temp from hidden state
                inv_temperature = torch.log(1 + torch.exp(self.gumbel_temperature(x))) + t0

                # sample the word using gumbel softmax
                z_one_hot = self.gumbel_softmax(x, inv_temperature, hard=True, is_prob=False)
            else:  # use argmax only for testing
                _, ind = F.softmax(x, dim=1).max(dim=1)
                y_hard = torch.zeros_like(x).view(-1, x.size(dim=1))
                y_hard.scatter_(1, ind.view(-1, 1), 1)
                z_one_hot = y_hard.view(x.size())

            return x, z_one_hot
        else:

            return x

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, inv_temperature, is_prob=True):
        if is_prob:
            y = torch.log(logits) + self.gumbel_sampler.sample(logits.size()).squeeze(2).cuda()  # sample from Gumbel distribution with loc=0, scale=1
        else:
            y = logits + self.gumbel_sampler.sample(logits.size()).squeeze(2).cuda()  # sample from Gumbel distribution with loc=0, scale=1
        return F.softmax(y * inv_temperature, dim=1)

    def gumbel_softmax(self, logits, inv_temperature, hard=False, is_prob=True):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, inv_temperature, is_prob)

        if not hard:
            return y.view(-1, self.num_words)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        # return y_hard.view(-1, y.size(1))
        return y_hard


# class Generator(nn.Module):
#     def __init__(self, nz=100):
#         super(Generator, self).__init__()
#         self.network = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(512, 256, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#
#             # state size. (ngf*2) x 8 x 8
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#
#             # state size. (ngf) x 16 x 16
#             nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 32 x 32
#         )
#
#         # nninit.kaiming_normal_(self.network..weight, nonlinearity='relu')
#         # nninit.kaiming_normal_(self.conv2_1.weight, nonlinearity='relu')
#         # nninit.kaiming_normal_(self.conv2_2.weight, nonlinearity='relu')
#         # nninit.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
#
#     def forward(self, x):
#         # print(x.shape)  # torch.Size([64, 100, 1, 1])
#         x = self.network(x)
#         # print(x.shape)  # torch.Size([64, 1, 28, 28])
#
#         return x


class Generator(nn.Module):
    def __init__(self, nz=100):
        super(Generator, self).__init__()
        # input is Z, going into a convolution
        self.conv1_1 = nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False)
        self.conv1_1_bn = nn.BatchNorm2d(512)

        # state size. (ngf*8) x 4 x 4
        self.conv2_1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, bias=False)
        self.conv2_1_bn = nn.BatchNorm2d(256)

        # state size. (ngf*2) x 8 x 8
        self.conv3_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.conv3_1_bn = nn.BatchNorm2d(128)

        # state size. (ngf) x 16 x 16
        self.conv4_1 = nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False)
        # state size. (nc) x 32 x 32

        # nninit.kaiming_normal_(self.conv1_1.weight, nonlinearity='relu')
        # nninit.kaiming_normal_(self.conv2_1.weight, nonlinearity='relu')
        # nninit.kaiming_normal_(self.conv3_1.weight, nonlinearity='relu')
        # nninit.xavier_normal_(self.conv4_1.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, x):
        # print(x.shape)  # torch.Size([64, 100, 1, 1])
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = torch.tanh(self.conv4_1(x))
        # print(x.shape)  # torch.Size([64, 1, 28, 28])

        return x

