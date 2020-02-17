import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = torch.mean(x, 1).view(x.size()[0], 1, x.size()[2], x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        #print(x.size())
        x = x.view(x.size(0), 48*4*4)
        return x


class Classifier(nn.Module):
    def __init__(self, args, prob=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(48*4*4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob
        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

    def forward(self, x):
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz=100):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(512, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, x):
        # print(x.shape)  # torch.Size([64, 100, 1, 1])
        x = self.network(x)
        # print(x.shape)  # torch.Size([64, 1, 28, 28])

        return x

