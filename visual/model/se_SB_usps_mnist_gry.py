import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nninit


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

        # self.drop1 = nn.Dropout()
        # self.fc3 = nn.Linear(1024, 256)
        # self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1024)
        # x = self.drop1(x)
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes=10):
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

        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout()
        self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x):
        # x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))
        #
        # x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        # x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        # x = x.view(-1, 1024)

        x = self.drop1(x)

        x = F.relu(self.fc3_bn(self.fc3(x)))
        # x = self.fc4(self.drop2(x))
        x = self.fc4(x)
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

