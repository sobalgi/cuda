import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # self.bn1 = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.bn2 = nn.BatchNorm2d(50)

        self.fc1 = nn.Linear(50*4*4, 500)

    def forward(self, x):
        # x = F.relu(self.bn1(self.pool1(self.conv1(x))))
        # x = F.relu(self.bn2(self.pool2(self.conv2(x))))
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(x.size(0), 50*4*4)
        x = self.fc1(x)
        return x


class Classifier(nn.Module):
    def __init__(self, args, n_classes=10):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(500, n_classes)
        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

    def forward(self, x):
        x = F.dropout(F.relu(x))
        x = self.fc2(x)
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

