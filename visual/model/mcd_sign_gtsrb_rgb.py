import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=2, padding=0)
        x = x.view(x.size(0), 6400)
        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(6400, 512)
        self.bn2_fc = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 43)
        self.bn_fc3 = nn.BatchNorm1d(43)
        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

    def forward(self, x):
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz=100):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 512, 5, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # state size. (ngf*2) x 10 x 10
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # state size. (ngf) x 20 x 20
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 40 x 40
        )

    def forward(self, x):
        # print(x.shape)  # torch.Size([64, 100, 1, 1])
        x = self.network(x)
        # print(x.shape)  # torch.Size([64, 3, 40, 40])

        return x
