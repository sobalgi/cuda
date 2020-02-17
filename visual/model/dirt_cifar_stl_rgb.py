import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nninit


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 96, (3, 3), stride=1, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(96)
        self.conv1_2 = nn.Conv2d(96, 96, (3, 3), stride=1, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(96)
        self.conv1_3 = nn.Conv2d(96, 96, (3, 3), stride=1, padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout(0.5)

        self.conv2_1 = nn.Conv2d(96, 192, (3, 3), stride=1, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(192)
        self.conv2_2 = nn.Conv2d(192, 192, (3, 3), stride=1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(192)
        self.conv2_3 = nn.Conv2d(192, 192, (3, 3), stride=1, padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout(0.5)

        self.conv3_1 = nn.Conv2d(192, 192, (3, 3), stride=1, padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 192, (1, 1), stride=1, padding=0)
        self.conv3_2_bn = nn.BatchNorm2d(192)
        self.conv3_3 = nn.Conv2d(192, 192, (1, 1), stride=1, padding=0)
        self.conv3_3_bn = nn.BatchNorm2d(192)
        self.drop3 = nn.Dropout(0.5)

        # nninit.xavier_normal(self.conv1_1.weight)
        # nninit.xavier_normal(self.conv1_2.weight)
        # nninit.xavier_normal(self.conv1_3.weight)
        # nninit.xavier_normal(self.conv2_1.weight)
        # nninit.xavier_normal(self.conv2_2.weight)
        # nninit.xavier_normal(self.conv2_3.weight)
        # nninit.xavier_normal(self.conv3_1.weight)
        # nninit.xavier_normal(self.conv3_2.weight)
        # nninit.xavier_normal(self.conv3_3.weight)
        # nninit.xavier_normal(self.fc4.weight)

        nn.utils.weight_norm(self.conv1_1, 'weight')
        nn.utils.weight_norm(self.conv1_2, 'weight')
        nn.utils.weight_norm(self.conv1_3, 'weight')
        nn.utils.weight_norm(self.conv2_1, 'weight')
        nn.utils.weight_norm(self.conv2_2, 'weight')
        nn.utils.weight_norm(self.conv2_3, 'weight')
        nn.utils.weight_norm(self.conv3_1, 'weight')
        nn.utils.weight_norm(self.conv3_2, 'weight')
        nn.utils.weight_norm(self.conv3_3, 'weight')
        # nn.utils.weight_norm(self.fc4, 'weight')

        self.global_pool = nn.AvgPool2d(6)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_1_bn(self.conv1_1(x)), negative_slope=.2)
        x = F.leaky_relu(self.conv1_2_bn(self.conv1_2(x)), negative_slope=.2)
        x = F.leaky_relu(self.conv1_3_bn(self.conv1_3(x)), negative_slope=.2)
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.leaky_relu(self.conv2_1_bn(self.conv2_1(x)), negative_slope=.2)
        x = F.leaky_relu(self.conv2_2_bn(self.conv2_2(x)), negative_slope=.2)
        x = F.leaky_relu(self.conv2_3_bn(self.conv2_3(x)), negative_slope=.2)
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.leaky_relu(self.conv3_1_bn(self.conv3_1(x)), negative_slope=.2)
        x = F.leaky_relu(self.conv3_2_bn(self.conv3_2(x)), negative_slope=.2)
        x = F.leaky_relu(self.conv3_3_bn(self.conv3_3(x)), negative_slope=.2)

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 192)

        # x = self.fc4(x)
        return x


class Classifier(nn.Module):
    def __init__(self, args, n_classes=9):
        super(Classifier, self).__init__()

        # self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        # self.conv1_1_bn = nn.BatchNorm2d(128)
        # self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        # self.conv1_2_bn = nn.BatchNorm2d(128)
        # self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        # self.conv1_3_bn = nn.BatchNorm2d(128)
        # self.pool1 = nn.MaxPool2d((2, 2))
        # self.drop1 = nn.Dropout()
        #
        # self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        # self.conv2_1_bn = nn.BatchNorm2d(256)
        # self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        # self.conv2_2_bn = nn.BatchNorm2d(256)
        # self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        # self.conv2_3_bn = nn.BatchNorm2d(256)
        # self.pool2 = nn.MaxPool2d((2, 2))
        # self.drop2 = nn.Dropout()
        #
        # self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        # self.conv3_1_bn = nn.BatchNorm2d(512)
        # self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        # self.nin3_2_bn = nn.BatchNorm2d(256)
        # self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        # self.nin3_3_bn = nn.BatchNorm2d(128)

        # self.drop1 = nn.Dropout()
        # self.fc3 = nn.Linear(128, 128)
        # self.fc3_bn = nn.BatchNorm1d(128)
        # self.drop2 = nn.Dropout()
        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        self.fc4 = nn.Linear(192, n_classes)
        nn.utils.weight_norm(self.fc4, 'weight')

    def forward(self, x):
        # x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        # x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        # x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        # x = self.drop1(x)
        #
        # x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        # x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        # x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        # x = self.drop2(x)
        #
        # x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        # x = F.relu(self.nin3_2_bn(self.nin3_2(x)))
        # x = F.relu(self.nin3_3_bn(self.nin3_3(x)))
        #
        # x = F.avg_pool2d(x, 6)
        # x = x.view(-1, 128)

        # x = self.drop1(x)
        # x = F.relu(self.fc3_bn(self.fc3(x)))
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
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, x):
        # print(x.shape)  # torch.Size([64, 100, 1, 1])
        x = self.network(x)
        # print(x.shape)  # torch.Size([64, 3, 32, 32])

        return x


class RGB_128_256_down_gp(nn.Module):
    def __init__(self, n_classes):
        super(RGB_128_256_down_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = nn.BatchNorm2d(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = nn.BatchNorm2d(128)

        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x)))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x)))

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)

        x = self.fc4(x)
        return x
