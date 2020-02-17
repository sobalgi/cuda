import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nninit


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(250 * 2 * 2, 350)
        # self.fc2 = nn.Linear(350, nclasses)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)), 2))
        x = self.conv_drop(x)
        x = x.view(-1, 250 * 2 * 2)
        return x
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)


class Classifier(nn.Module):
    def __init__(self, args, n_classes=10):
        super(Classifier, self).__init__()

        # self.conv1_1 = nn.Conv2d(3, 96, (3, 3), padding=1)
        # self.conv1_1_bn = nn.BatchNorm2d(96)
        # self.conv1_2 = nn.Conv2d(96, 96, (3, 3), padding=1)
        # self.conv1_2_bn = nn.BatchNorm2d(96)
        # self.conv1_3 = nn.Conv2d(96, 96, (3, 3), padding=1)
        # self.conv1_3_bn = nn.BatchNorm2d(96)
        # self.pool1 = nn.MaxPool2d((2, 2))
        #
        # self.conv2_1 = nn.Conv2d(96, 192, (3, 3), padding=1)
        # self.conv2_1_bn = nn.BatchNorm2d(192)
        # self.conv2_2 = nn.Conv2d(192, 192, (3, 3), padding=1)
        # self.conv2_2_bn = nn.BatchNorm2d(192)
        # self.conv2_3 = nn.Conv2d(192, 192, (3, 3), padding=1)
        # self.conv2_3_bn = nn.BatchNorm2d(192)
        # self.pool2 = nn.MaxPool2d((2, 2))
        #
        # self.conv3_1 = nn.Conv2d(192, 384, (3, 3), padding=1)
        # self.conv3_1_bn = nn.BatchNorm2d(384)
        # self.conv3_2 = nn.Conv2d(384, 384, (3, 3), padding=1)
        # self.conv3_2_bn = nn.BatchNorm2d(384)
        # self.conv3_3 = nn.Conv2d(384, 384, (3, 3), padding=1)
        # self.conv3_3_bn = nn.BatchNorm2d(384)
        # self.pool3 = nn.MaxPool2d((2, 2))
        #
        # self.drop1 = nn.Dropout()

        # self.fc3 = nn.Linear(384, 128)
        # self.fc3_bn = nn.BatchNorm1d(128)

        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        self.fc1 = nn.Linear(250 * 2 * 2, 350)
        self.fc2 = nn.Linear(350, n_classes)

    def forward(self, x):
        # x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        # x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        # x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        #
        # x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        # x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        # x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        #
        # x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        # x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        # x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x))))
        #
        # x = self.drop1(x)
        # x = F.avg_pool2d(x, 5)
        # x = x.view(-1, 384)

        # x = self.drop1(x)
        # x = F.relu(self.fc3_bn(self.fc3(x)))
        # x = self.fc4(self.drop2(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
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


# Overfitts easily on synsigns-gtsrb