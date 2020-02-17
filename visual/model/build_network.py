import torch
import torch.nn as nn
import torch.nn.init as nninit
from torch.autograd import Variable
import torch.optim as optim
import os
import tensorboardX
from tensorboardX import SummaryWriter


def weight_init_general(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nninit.normal_(m.weight.data)
        if m.bias is not None:
            nninit.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nninit.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nninit.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nninit.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nninit.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nninit.normal_(m.weight.data)
        if m.bias is not None:
            nninit.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nninit.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nninit.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nninit.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nninit.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nninit.normal_(m.weight.data, mean=1, std=0.02)
        nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nninit.normal_(m.weight.data, mean=1, std=0.02)
        nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nninit.normal_(m.weight.data, mean=1, std=0.02)
        nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nninit.xavier_normal_(m.weight.data)
        nninit.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nninit.orthogonal_(param.data)
            else:
                nninit.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nninit.orthogonal_(param.data)
            else:
                nninit.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nninit.orthogonal_(param.data)
            else:
                nninit.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nninit.orthogonal_(param.data)
            else:
                nninit.normal_(param.data)


def weight_init_se(m):
    ''' nninit layer parameters. '''
    if isinstance(m, nn.Conv2d):
        nninit.kaiming_normal_(m.weight.data, mode='fan_out')
        if m.bias is not None:
            nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nninit.constant_(m.weight.data, 1)
        nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nninit.normal_(m.weight.data, std=1e-3)
        if m.bias is not None:
            nninit.constant_(m.bias.data, 0)


def weight_init_dade(m):
    ''' nninit layer parameters. '''
    if isinstance(m, nn.Conv2d):
        nninit.normal_(m.weight.data, mean=0.0, std=0.05)
        if m.bias is not None:
            nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nninit.normal_(m.weight.data, mean=1.0, std=0.02)
        nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nninit.normal_(m.weight.data, mean=0.0, std=0.05)
        if m.bias is not None:
            nninit.constant_(m.bias.data, 0)


def weight_init_xavier(m):
    ''' nninit layer parameters. '''
    if isinstance(m, nn.Conv2d):
        nninit.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            nninit.constant_(m.bias.data, 0)


def weight_init_mmd(m):
    ''' nninit layer parameters. '''
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            nninit.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            nninit.constant_(m.bias.data, 0)


class DEVILDA(nn.Module):
    def __init__(self, Encoder, Classifier, Generator):
        super(DEVILDA, self).__init__()
        # E = net_arch.Encoder()
        # C = net_arch.Classifier(args)
        # GS = net_arch.Generator(nz=args.nz)
        # GT = net_arch.Generator(args.nz)

        self.E = Encoder
        self.C = Classifier
        self.GS = Generator

    def forward(self, x):
        x = self.GS(x)
        x = self.E(x)
        x = self.C(x)

        # Add the Graphs to tensorboard logs
        # E_in = torch.zeros(1, args.nc, args.image_size, args.image_size)
        # self.writer.add_graph(net_arch.Encoder(), (E_in,), True)
        # C_in = E(torch.zeros(1, args.nc, args.image_size, args.image_size), )
        # self.writer.add_graph(net_arch.Classifier(args), (C_in,), True)
        # G_in = torch.zeros(1, args.nz, 1, 1)
        # self.writer.add_graph(net_arch.Generator(nz=args.nz), (G_in,), True)
        return x


class Network(object):
    def __init__(self, args):

        log_dir = args.ckpt_dir
        # if tensorboardX.__version__ == '1.7':
        #     self.writer = SummaryWriter(logdir=log_dir)
        #     ckpt_dir = os.path.join(args.absolute_base_path, self.writer.logdir)
        # else:
        #     self.writer = SummaryWriter(log_dir=log_dir)
        #     ckpt_dir = os.path.join(args.absolute_base_path, self.writer.log_dir)

        # self.writer = SummaryWriter(logdir=log_dir)
        # ckpt_dir = os.path.join(args.absolute_base_path, self.writer.logdir)
        self.writer = SummaryWriter(log_dir=log_dir)
        ckpt_dir = os.path.join(args.absolute_base_path, self.writer.log_dir)
        args.ckpt_dir = ckpt_dir

        # args.use_gumbel = True
        self.args = args
        self.exp = args.exp
        self.last_run_dir = './last_run/' + self.args.exp + '_' + self.args.epoch_size
        self.model_dir = self.args.ckpt_dir  # + '/Model'

        if args.exp in {'mnist_usps', 'usps_mnist'}:  # (1, 28, 28)
            if args.network_type == 'se':
                from model import se_usps_mnist_gry as net_arch
            elif args.network_type == 'dade':
                from model import dade_usps_mnist_gry as net_arch
            elif args.network_type == 'mcd':
                from model import mcd_usps_mnist_gry as net_arch
            else:
                raise NotImplementedError
        elif args.exp in {'svhn_mnist', 'mnist_svhn'}:  # (3, 32, 32)
            if args.network_type == 'se':
                if args.nc == 1:
                    # from model import se_usps_mnist_gry as net_arch
                    from model import se_svhn_mnist_gry as net_arch
                else:
                    from model import se_svhn_mnist_rgb as net_arch
            elif args.network_type == 'dade':
                from model import dade_usps_mnist_gry as net_arch
            elif args.network_type == 'stn':
                if args.nc == 1:
                    # from model import se_usps_mnist_gry as net_arch
                    from model import stn_svhn_mnist_gry as net_arch
                else:
                    from model import stn_svhn_mnist_rgb as net_arch
            elif args.network_type == 'mcd':
                from model import mcd_svhn_mnist_rgb as net_arch
            else:
                raise NotImplementedError
        elif args.exp in {'syndigits_svhn', 'svhn_syndigits'}:  # (3, 32, 32)
            if args.network_type == 'se':
                if args.nc == 1:
                    # from model import se_usps_mnist_gry as net_arch
                    from model import se_svhn_synnum_gry as net_arch
                else:
                    from model import se_svhn_synnum_rgb as net_arch
            elif args.network_type == 'stn':
                from model import stn_syndigits_svhn_rgb as net_arch
            elif args.network_type == 'dade':
                from model import dade_usps_mnist_gry as net_arch
            elif args.network_type == 'mcd':
                from model import mcd_svhn_synnum_rgb as net_arch
            else:
                raise NotImplementedError
        elif args.exp in {'cifar_stl', 'stl_cifar'}:  # (3, 32, 32)
            if args.network_type == 'se':
                from model import se_cifar_stl_rgb as net_arch
            elif args.network_type == 'dpn':
                from model import dpn_cifar_stl_rgb as net_arch
            elif args.network_type == 'dade':
                from model import dade_cifar_stl_rgb as net_arch
            elif args.network_type == 'dirt':
                from model import dirt_cifar_stl_rgb as net_arch
            elif args.network_type == 'mcd':
                from model import mcd_cifar_stl_rgb as net_arch
            else:
                raise NotImplementedError
        elif args.exp in {'synsigns_gtsrb', 'gtsrb_synsigns'}:  # (3, 40, 40)
            if args.network_type == 'se':
                from model import se_sign_gtsrb_rgb as net_arch
            elif args.network_type == 'stn':
                from model import stn_sign_gtsrb_rgb as net_arch
            elif args.network_type == 'dade':
                from model import dade_sign_gtsrb_rgb as net_arch
            elif args.network_type == 'mcd':
                from model import mcd_sign_gtsrb_rgb as net_arch
            else:
                raise NotImplementedError
        elif args.exp in {'amazon_dslr', 'amazon_webcam',
                          'dslr_amazon', 'dslr_webcam',
                          'webcam_amazon', 'webcam_dslr'}:  # (3, 224, 224)
            if args.network_type == 'resnet':
                from model import resnet_office_rgb as net_arch
            elif args.network_type == 'cdan':
                from model import cdan_btl_office_rgb as net_arch
            # elif args.network_type == 'dade':
            #     from model import dade_sign_gtsrb_rgb as net_arch
            # elif args.network_type == 'mcd':
            #     from model import mcd_sign_gtsrb_rgb as net_arch
            else:
                raise NotImplementedError
        else:
            raise ValueError('Unknown experiment type \'{}\''.format(args.exp))

        E = net_arch.Encoder()
        C = net_arch.Classifier(args)
        GS = net_arch.Generator(nz=args.nz)
        # GT = net_arch.Generator(args.nz)

        # Add the Graphs to tensorboard logs
        # E_in = torch.zeros(1, args.nc, args.image_size, args.image_size)
        # self.writer.add_graph(net_arch.Encoder(), (E_in,), True)
        # C_in = E(torch.zeros(1, args.nc, args.image_size, args.image_size), )
        # self.writer.add_graph(net_arch.Classifier(args), (C_in,), True)
        # G_in = torch.zeros(1, args.nz, 1, 1)
        # self.writer.add_graph(net_arch.Generator(nz=args.nz), (G_in,), True)
        G_in = torch.zeros(1, args.nz, 1, 1)
        try:
            self.writer.add_graph(DEVILDA(E, C, GS), (G_in,), True)
        except:
            #G_in = torch.zeros(1, args.nz, 1, 1)
            #self.writer.add_graph(net_arch.Generator(nz=args.nz), (G_in,), True)
            #E_in = torch.zeros(1, args.nc, args.image_size, args.image_size)
            #self.writer.add_graph(net_arch.Encoder(), (E_in,), True)
            # C_in = E(torch.zeros(1, args.nc, args.image_size, args.image_size), )
            # self.writer.add_graph(net_arch.Classifier(args), (C_in,), True)
            pass

        if args.use_tied_gen:
            # GS.network.weight.data = GT.network.weight.data
            GT = GS
        else:
            GT = net_arch.Generator(nz=args.nz)

        self.num_gpus = len(args.gpus.split(','))
        self.gpus = list(range(torch.cuda.device_count()))

        E = torch.nn.DataParallel(E, device_ids=self.gpus)
        C = torch.nn.DataParallel(C, device_ids=self.gpus)
        GS = torch.nn.DataParallel(GS, device_ids=self.gpus)
        GT = torch.nn.DataParallel(GT, device_ids=self.gpus)

        if torch.cuda.is_available():
            E.cuda()
            C.cuda()
            GS.cuda()
            GT.cuda()

        self.E = E
        self.C = C
        self.GS = GS
        self.GT = GT

        if args.weight_init == 'weight_init_general':
            weight_init = weight_init_general
        elif args.weight_init == 'weight_init_se':
            weight_init = weight_init_se
        elif args.weight_init == 'weight_init_xavier':
            weight_init = weight_init_xavier
        elif args.weight_init == 'weight_init_mmd':
            weight_init = weight_init_mmd
        elif args.weight_init == 'weight_init_dade':
            weight_init = weight_init_dade
        else:
            weight_init = None

        if weight_init:
            self.E.apply(weight_init)
            self.C.apply(weight_init)
            self.GS.apply(weight_init)
            self.GT.apply(weight_init)
        else:
            # initialize the network
            # self.load_model(args.load_checkpoint)
            pass

        self.optimizerE = optim.Adam(self.E.parameters(),
                                    lr=args.learning_rate*0.01,
                                    betas=(0.5, 0.999))
        self.optimizerC = optim.Adam(self.C.parameters(),
                                    lr=args.learning_rate,
                                    betas=(0.5, 0.999))
        self.optimizerGS = optim.Adam(self.GS.parameters(),
                                      lr=args.learning_rate,
                                      betas=(0.5, 0.999))
        self.optimizerGT = optim.Adam(self.GT.parameters(),
                                      lr=args.learning_rate,
                                      betas=(0.5, 0.999))

        # self.optimizerE = optim.SGD(self.E.parameters(),
        #                             lr=args.learning_rate*.001,
        #                             momentum=0.9)
        # self.optimizerC = optim.SGD(self.C.parameters(),
        #                             lr=args.learning_rate,
        #                             momentum=0.9)
        # self.optimizerE = optim.SGD(self.E.module.get_parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
        # self.optimizerC = optim.SGD(self.C.module.get_parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
        # self.optimizerGS = optim.SGD(self.GS.parameters(),
        #                              lr=args.learning_rate,
        #                              momentum=0.9)
        # self.optimizerGT = optim.SGD(self.GT.parameters(),
        #                              lr=args.learning_rate,
        #                              momentum=0.9)

        if torch.cuda.is_available():
            self.fixed_noise = torch.FloatTensor(self.args.batch_size, self.args.nz, 1, 1).normal_(0, 1).cuda()
            self.one = torch.FloatTensor([1]).cuda()
            self.mone = self.one * -1

    def zero_grad(self):
        self.E.zero_grad()
        self.C.zero_grad()
        self.GS.zero_grad()
        self.GT.zero_grad()
        self.optimizerE.zero_grad()
        self.optimizerC.zero_grad()
        self.optimizerGS.zero_grad()
        self.optimizerGT.zero_grad()

    def save_model(self, suffix=''):
        os.system('mkdir -p {}/Model'.format(self.last_run_dir))
        os.system('mkdir -p {}/Model'.format(self.model_dir))

        torch.save(self.E.module.state_dict(), '%s/Model/netE%s.pth' % (self.last_run_dir, suffix))
        torch.save(self.C.module.state_dict(), '%s/Model/netC%s.pth' % (self.last_run_dir, suffix))
        torch.save(self.GS.module.state_dict(), '%s/Model/netGS%s.pth' % (self.last_run_dir, suffix))
        torch.save(self.GT.module.state_dict(), '%s/Model/netGT%s.pth' % (self.last_run_dir, suffix))
        torch.save(self.E.module.state_dict(), '%s/Model/netE%s.pth' % (self.model_dir, suffix))
        torch.save(self.C.module.state_dict(), '%s/Model/netC%s.pth' % (self.model_dir, suffix))
        torch.save(self.GS.module.state_dict(), '%s/Model/netGS%s.pth' % (self.model_dir, suffix))
        torch.save(self.GT.module.state_dict(), '%s/Model/netGT%s.pth' % (self.model_dir, suffix))

        # print('E/C/GS/GT models saved.')

    def load_model(self, load_checkpoint='', suffix=''):
        if load_checkpoint == '':
            load_checkpoint = self.last_run_dir

        try:
            self.E.module.load_state_dict(torch.load('%s/Model/netE%s.pth' % (load_checkpoint, suffix)))
            self.C.module.load_state_dict(torch.load('%s/Model/netC%s.pth' % (load_checkpoint, suffix)))
            self.GS.module.load_state_dict(torch.load('%s/Model/netGS%s.pth' % (load_checkpoint, suffix)))
            self.GT.module.load_state_dict(torch.load('%s/Model/netGT%s.pth' % (load_checkpoint, suffix)))
            print('Loaded E/C/GS/GT models from {0} .'.format(load_checkpoint))
        except:
            print('Error while loading net_E/C/GS/GT models from {0} !!!'.format(load_checkpoint))

    def discriminator(self, X):
        feature = self.E(X)
        if self.C.module.use_gumbel:
            logits, label_one_hot = self.C.forward(feature)
            return feature, logits, label_one_hot
        else:
            logits = self.C.forward(feature)
            return feature, logits
