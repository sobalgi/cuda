import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nninit
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, modelpath="./pretrained_model/resnet152-b121ed2d.pth"):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Encoder, self).__init__()

        #resnet = models.resnet152(pretrained=True)
        pretrained = False
        resnet = models.resnet152(pretrained)
        #
        checkpoint = torch.load(modelpath)
        resnet.load_state_dict(checkpoint)

        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        # self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        # self.init_weigths()

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 2048)

        # x = self.fc4(x)
        return x


class Classifier(nn.Module):
    def __init__(self, args, n_classes=31):
        super(Classifier, self).__init__()

        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        self.fc5 = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.fc5(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz=100, n_ch=3, image_dim=224, ngf=64):
        super(Generator, self).__init__()
        assert image_dim % 16 == 0, "image_dim has to be a multiple of 16"

        cngf, timage_dim = ngf // 2, 4
        while timage_dim < image_dim:
            cngf = cngf * 2
            timage_dim = timage_dim * 2

        main = nn.Sequential()
        main.add_module('initial_{0}_{1}_convt'.format(nz, cngf), nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_elu'.format(cngf), nn.ELU())

        csize = 4
        while csize < image_dim // 2:
            main.add_module('pyramid_{0}_{1}_convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid_{0}_elu'.format(cngf // 2),
                            nn.ELU())
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final_{0}_{1}_convt'.format(cngf, n_ch), nn.ConvTranspose2d(cngf, n_ch, 4, 2, 1, bias=False))
        main.add_module('final_{0}_{1}_adapavgpool2d'.format(image_dim, image_dim), nn.AdaptiveAvgPool2d((image_dim, image_dim)))  # change image to batchsize x n_ch x image_dim x image_dim
        main.add_module('final_{0}_tanh'.format(n_ch),
                        nn.Tanh())

        self.network = main

    def forward(self, x):
        # print(x.shape)  # torch.Size([64, 100, 1, 1])
        x = self.network(x)
        # print(x.shape)  # torch.Size([64, 3, 32, 32])

        return x
