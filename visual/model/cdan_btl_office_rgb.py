import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nninit
import torchvision.models as models

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class ResNet50_office(nn.Module):
  def __init__(self, resnet_name='ResNet50', use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000, modelpath="./pretrained_model/resnet50-19c8e357.pth"):
    super(ResNet50_office, self).__init__()
    # resnet = models.resnet152(pretrained=True)
    pretrained = False
    # model_resnet = models.resnet152(pretrained)
    model_resnet = resnet_dict[resnet_name](pretrained=pretrained)
    #
    checkpoint = torch.load(modelpath)
    model_resnet.load_state_dict(checkpoint)

    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


class Encoder(nn.Module):
  def __init__(self, resnet_name='ResNet50', use_bottleneck=False, bottleneck_dim=256, new_cls=True, class_num=1000, modelpath="./pretrained_model/resnet50-19c8e357.pth"):
    super(Encoder, self).__init__()
    # resnet = models.resnet152(pretrained=True)
    pretrained = False
    model_resnet = models.resnet50(pretrained)
    # model_resnet = resnet_dict[resnet_name](pretrained=pretrained)
    #
    checkpoint = torch.load(modelpath)
    model_resnet.load_state_dict(checkpoint)

    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            # self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            # self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            # self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            # self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        #self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    # y = self.fc(x)
    return x # , y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            # {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}
                              ]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            # {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}
                              ]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


class Classifier(nn.Module):
  def __init__(self, args, resnet_name='ResNet50', use_bottleneck=False, bottleneck_dim=256, new_cls=True, class_num=31):
    super(Classifier, self).__init__()
    pretrained = False
    model_resnet = models.resnet50(pretrained)
    # model_resnet = resnet_dict[resnet_name](pretrained=pretrained)
    # model_resnet = resnet_dict[resnet_name](pretrained=False)
    # self.conv1 = model_resnet.conv1
    # self.bn1 = model_resnet.bn1
    # self.relu = model_resnet.relu
    # self.maxpool = model_resnet.maxpool
    # self.layer1 = model_resnet.layer1
    # self.layer2 = model_resnet.layer2
    # self.layer3 = model_resnet.layer3
    # self.layer4 = model_resnet.layer4
    # self.avgpool = model_resnet.avgpool
    # self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
    #                      self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    #
    self.use_drop = False
    self.use_bn = False
    self.use_gumbel = False 
    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            # self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            # self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    # x = self.feature_layers(x)
    # x = x.view(x.size(0), -1)
    # if self.use_bottleneck and self.new_cls:
    #     x = self.bottleneck(x)
    y = self.fc(x)
    return y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [
                # {"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                # {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}
            ]
        else:
            parameter_list = [
                # {"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}
            ]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


# class Encoder(nn.Module):
#     def __init__(self, modelpath="./pretrained_model/resnet50-19c8e357.pth"):
#         """Load the pretrained ResNet-152 and replace top fc layer.""" #resnet152-b121ed2d.pth
#         super(Encoder, self).__init__()
#
#         #resnet = models.resnet152(pretrained=True)
#         pretrained = False
#         resnet = models.resnet152(pretrained)
#         #
#         checkpoint = torch.load(modelpath)
#         resnet.load_state_dict(checkpoint)
#
#         modules = list(resnet.children())[:-1]  # delete the last fc layer.
#         self.resnet = nn.Sequential(*modules)
#         # self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
#         # self.init_weigths()
#
#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.view(-1, 2048)
#
#         # x = self.fc4(x)
#         return x


# class Classifier(nn.Module):
#     def __init__(self, args, n_classes=31):
#         super(Classifier, self).__init__()
#
#         self.use_drop = args.use_drop
#         self.use_bn = args.use_bn
#         self.use_gumbel = args.use_gumbel
#
#         self.fc5 = nn.Linear(2048, n_classes)
#
#     def forward(self, x):
#         x = self.fc5(x)
#         return x


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
