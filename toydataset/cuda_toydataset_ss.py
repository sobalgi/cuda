import torch
import torch.cuda
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dutils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
# matplotlib.use('WXAgg')
# matplotlib.use('GTKAgg')
# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

# from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
# from sklearn.datasets import make_gaussian_quantiles

import numpy as np
import random
import os
import sys
import socket
import datetime
import math
import timeit
import cmdline_helpers
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import torch
import torch.cuda
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import dataset
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class Encoder(nn.Module):
    def __init__(self, args, output_dim):
        super(Encoder, self).__init__()
        self.fc3 = nn.Linear(args.input_dim, 6)
        # self.fc3 = nn.Linear(8, 6)

        self.fc4 = nn.Linear(6, 4)

        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        if self.use_drop:
            self.drop1 = nn.Dropout()
            # self.drop1 = nn.AlphaDropout(p=0.2)

        if self.use_bn:
            self.fc3_bn = nn.BatchNorm1d(output_dim)

        self.act = nn.SELU()

    def forward(self, x):
        # if self.use_drop:
        #     x = self.drop1(x)
        #
        if self.use_bn:
            x = torch.sigmoid(self.fc3_bn(self.fc3(x)))
        else:
            x = torch.sigmoid(self.fc3(x))
            x = torch.sigmoid(self.fc4(self.drop1(x)))
            # x = self.act(self.fc3(x))
            # x = self.act(self.fc4(self.drop1(x)))

        return x


class Classifier(nn.Module):
    def __init__(self, args, input_dim, n_classes=2):
        super(Classifier, self).__init__()
        self.use_drop = args.use_drop
        self.use_bn = args.use_bn
        self.use_gumbel = args.use_gumbel

        if self.use_drop:
            self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        if self.use_drop:
            x = self.drop1(x)
        x = self.fc4(x)  # send logits as out
        return x


class DEVILDA(nn.Module):
    def __init__(self, Encoder, Classifier):
        super(DEVILDA, self).__init__()
        # E = net_arch.Encoder()
        # C = net_arch.Classifier(args)
        # GS = net_arch.Generator(nz=args.nz)
        # GT = net_arch.Generator(args.nz)

        self.E = Encoder
        self.C = Classifier

    def forward(self, x):
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
        self.writer = SummaryWriter(logdir=log_dir)
        # self.writer = SummaryWriter(log_dir=log_dir)  # old tbx

        ckpt_dir = os.path.join(args.absolute_base_path, self.writer.logdir)
        # ckpt_dir = os.path.join(args.absolute_base_path, self.writer.log_dir)  # old tbx
        args.ckpt_dir = ckpt_dir

        # args.use_gumbel = True
        self.args = args
        self.exp = args.exp
        self.last_run_dir = './last_run/' + self.args.exp + '_' + self.args.epoch_size
        self.model_dir = self.args.ckpt_dir  # + '/Model'

        E = Encoder(args, output_dim=6)
        C = Classifier(args, input_dim=4)
        # GT = net_arch.Generator(args.nz)

        # Add the Graphs to tensorboard logs
        # E_in = torch.zeros(1, args.nc, args.image_size, args.image_size)
        # self.writer.add_graph(net_arch.Encoder(), (E_in,), True)
        # C_in = E(torch.zeros(1, args.nc, args.image_size, args.image_size), )
        # self.writer.add_graph(net_arch.Classifier(args), (C_in,), True)
        # G_in = torch.zeros(1, args.nz, 1, 1)
        # self.writer.add_graph(net_arch.Generator(nz=args.nz), (G_in,), True)
        E_in = torch.zeros(1, args.input_dim)
        try:
            self.writer.add_graph(DEVILDA(E, C), (E_in,), True)
        except:
            #G_in = torch.zeros(1, args.nz, 1, 1)
            #self.writer.add_graph(net_arch.Generator(nz=args.nz), (G_in,), True)
            #E_in = torch.zeros(1, args.nc, args.image_size, args.image_size)
            #self.writer.add_graph(net_arch.Encoder(), (E_in,), True)
            # C_in = E(torch.zeros(1, args.nc, args.image_size, args.image_size), )
            # self.writer.add_graph(net_arch.Classifier(args), (E_in,), True)
            pass

        self.num_gpus = len(args.gpus.split(','))
        self.gpus = list(range(torch.cuda.device_count()))

        E = torch.nn.DataParallel(E, device_ids=self.gpus)
        C = torch.nn.DataParallel(C, device_ids=self.gpus)
        # D = torch.nn.DataParallel(C, device_ids=self.gpus)

        if torch.cuda.is_available():
            E.cuda()
            C.cuda()
            # D.cuda()

        self.E = E
        self.C = C
        # self.D = C

        # if args.weight_init == 'weight_init_general':
        #     weight_init = weight_init_general
        # elif args.weight_init == 'weight_init_se':
        #     weight_init = weight_init_se
        # elif args.weight_init == 'weight_init_xavier':
        #     weight_init = weight_init_xavier
        # elif args.weight_init == 'weight_init_mmd':
        #     weight_init = weight_init_mmd
        # elif args.weight_init == 'weight_init_dade':
        #     weight_init = weight_init_dade
        # else:
        #     weight_init = None

        weight_init = None
        if weight_init:
            self.E.apply(weight_init)
            self.C.apply(weight_init)
            # self.D.apply(weight_init)
        else:
            # initialize the network
            # self.load_model(args.load_checkpoint)
            pass

        self.optimizerE = optim.Adam(self.E.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.5, 0.999))
        self.optimizerC = optim.Adam(self.C.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.5, 0.999))
        # self.optimizerD = optim.Adam(self.D.parameters(),
        #                              lr=args.learning_rate,
        #                              betas=(0.5, 0.999))

        # self.optimizerE = optim.SGD(self.E.parameters(),
        #                             lr=args.learning_rate,
        #                             momentum=0.9)
        # self.optimizerC = optim.SGD(self.C.parameters(),
        #                             lr=args.learning_rate,
        #                             momentum=0.9)
        # self.optimizerGS = optim.SGD(self.GS.parameters(),
        #                              lr=args.learning_rate,
        #                              momentum=0.9)
        # self.optimizerGT = optim.SGD(self.GT.parameters(),
        #                              lr=args.learning_rate,
        #                              momentum=0.9)

        self.one = torch.FloatTensor([1]).cuda()
        self.mone = self.one * -1

    def zero_grad(self):
        self.E.zero_grad()
        self.C.zero_grad()
        # self.D.zero_grad()
        self.optimizerE.zero_grad()
        self.optimizerC.zero_grad()
        # self.optimizerD.zero_grad()

    def save_model(self, suffix=''):
        os.system('mkdir -p {}/Model'.format(self.last_run_dir))
        os.system('mkdir -p {}/Model'.format(self.model_dir))

        torch.save(self.E.module.state_dict(), '%s/Model/netE%s.pth' % (self.last_run_dir, suffix))
        torch.save(self.C.module.state_dict(), '%s/Model/netC%s.pth' % (self.last_run_dir, suffix))
        # torch.save(self.D.module.state_dict(), '%s/Model/netD%s.pth' % (self.last_run_dir, suffix))
        torch.save(self.E.module.state_dict(), '%s/Model/netE%s.pth' % (self.model_dir, suffix))
        torch.save(self.C.module.state_dict(), '%s/Model/netC%s.pth' % (self.model_dir, suffix))
        # torch.save(self.D.module.state_dict(), '%s/Model/netD%s.pth' % (self.model_dir, suffix))

        # print('E/C/GS/GT models saved.')

    def load_model(self, load_checkpoint='', suffix=''):
        if load_checkpoint == '':
            load_checkpoint = self.last_run_dir

        try:
            self.E.module.load_state_dict(torch.load('%s/Model/netE%s.pth' % (load_checkpoint, suffix)))
            self.C.module.load_state_dict(torch.load('%s/Model/netC%s.pth' % (load_checkpoint, suffix)))
            # self.D.module.load_state_dict(torch.load('%s/Model/netD%s.pth' % (load_checkpoint, suffix)))
            print('Loaded E/C/D models from {0} .'.format(load_checkpoint))
        except:
            print('Error while loading net_E/C/D models from {0} !!!'.format(load_checkpoint))

    def discriminator(self, X):
        feature = self.E(X)
        if self.C.module.use_gumbel:
            C_logits, C_label_one_hot = self.C.forward(feature)
            # D_logits, D_label_one_hot = self.D.forward(feature)

            return feature, C_logits, C_label_one_hot
            # return feature, C_logits, C_label_one_hot, D_logits, D_label_one_hot
        else:
            C_logits = self.C.forward(feature)
            # D_logits = self.D.forward(feature)
            return feature, C_logits
            # return feature, C_logits, D_logits


def test(net, l_src_test, l_tgt_test):
    # discriminator.eval()
    net.E.eval()
    net.C.eval()
    # net.D.eval()
    # l_src_test = dutils.DataLoader(d_src_test,
    #                                batch_size=128,
    #                                shuffle=True,
    #                                num_workers=int(4),
    #                                drop_last=False)
    #
    # l_tgt_test = dutils.DataLoader(d_tgt_test,
    #                                batch_size=128,
    #                                shuffle=True,
    #                                num_workers=int(4),
    #                                drop_last=False)
    src_test_correct = 0
    tgt_test_correct = 0
    with torch.no_grad():
        for data, target in l_src_test:
            # data, target = data.to(device), target.to(device)
            data, target = data.cuda(), target.cuda()
            if net.C.module.use_gumbel:
                _, output_logits, output = net.discriminator(data)
            else:
                _, output_logits = net.discriminator(data)
                output = F.softmax(output_logits, dim=1)
            src_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            src_test_correct += src_test_pred.eq(target.view_as(src_test_pred)).sum().item()

        for data, target in l_tgt_test:
            # data, target = data.to(device), target.to(device)
            data, target = data.cuda(), target.cuda()
            if net.C.module.use_gumbel:
                _, output_logits, output = net.discriminator(data)
            else:
                _, output_logits = net.discriminator(data)
                output = F.softmax(output_logits, dim=1)
            tgt_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            tgt_test_correct += tgt_test_pred.eq(target.view_as(tgt_test_pred)).sum().item()

    # print('\nSRC Test Acc: {:.0f}%, TGT Test Acc: {:.0f}%\n'.format(
    #     100. * src_test_correct / len(src_test_loader.dataset),
    #     100. * tgt_test_correct / len(tgt_test_loader.dataset)))
    # return 100. * src_test_correct / len(src_test_loader.dataset), 100. * tgt_test_correct / len(tgt_test_loader.dataset)
    return src_test_correct, tgt_test_correct


def experiment(args):
    exp = args.exp
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    epoch_size = args.epoch_size
    # seed = 310, 1211, 1511, 2082, 2234, 211, 2737, 22,
    if args.seed == 0:
        seed = np.random.randint(5000)
        args.seed = seed
    else:
        seed = args.seed
    log_file = args.log_file
    workers = args.workers
    use_gpu = True
    gpus = args.gpus
    logs_dir = args.logs_dir
    ckpt_dir = args.ckpt_dir
    input_dim = args.input_dim
    n_classes = args.n_classes
    n_samples = args.n_samples
    split_size = args.split_size
    load_checkpoint = args.load_checkpoint
    use_sampler = args.use_sampler

    src_name = 'toy_d0'
    tgt_name = 'toy_d1'
    if not args.swap_domain:
        # X_src, y_src = make_blobs(n_samples=n_samples, n_features=input_dim, centers=2,
        #                           random_state=int(seed * 10))
        # X_tgt, y_tgt = make_blobs(n_samples=n_samples, n_features=input_dim, centers=2,
        #                           random_state=int(seed * 20))
        src_name = 'toy_d0'
        tgt_name = 'toy_d1'
    else:
        # X_tgt, y_tgt = make_blobs(n_samples=n_samples, n_features=input_dim, centers=2,
        #                           random_state=int(seed * 10))
        # X_src, y_src = make_blobs(n_samples=n_samples, n_features=input_dim, centers=2,
        #                           random_state=int(seed * 20))
        src_name = 'toy_d1'
        tgt_name = 'toy_d0'

    args.exp = src_name

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Some variable hardcoding - Only for development
    # seed = 1126
    seed = args.seed
    simul_train_src_tgt = True
    # use_scheduler = args.use_scheduler  # True/False
    use_sampler = args.use_sampler  # True/False
    use_ramp_sup = args.ramp > 0
    use_ramp_unsup = args.ramp > 0
    ramp_sup_weight_in_list = [1.0]
    ramp_unsup_weight_in_list = [1.0]

    lambda_tsl = 0.0
    lambda_ssl = 1.0
    lambda_sul = 0.0
    lambda_tul = 0.0
    lambda_sal = 0.0
    lambda_tal = 0.0

    machinename = socket.gethostname()
    # hostname = timestamp = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    # hostname = timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
    hostname = timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    absolute_pyfile_path = os.path.abspath(sys.argv[0])
    args.absolute_pyfile_path = os.path.abspath(sys.argv[0])

    absolute_base_path = os.path.dirname(absolute_pyfile_path)
    args.absolute_base_path = os.path.dirname(absolute_pyfile_path)

    # args.dataroot = os.path.expanduser(args.dataroot)
    # dataroot = os.path.join(absolute_base_path, args.dataroot)

    args.logs_dir = os.path.expanduser(args.logs_dir)
    # logs_dir = args.logs_dir
    logs_dir = os.path.join(absolute_base_path, args.logs_dir)
    args.logs_dir = logs_dir

    log_num = seed
    log_file = logs_dir + '/' + hostname + '_' + machinename + '_' + args.exp + '_' + str(log_num) + '_' + args.epoch_size + '_ss' + '.txt'

    # Setup logfile to store output logs
    if log_file is not None:
        while os.path.exists(log_file):
            log_num += 1
            log_file = '{0}/{3}_{1}_{2}_{4}_ss.txt'.format(args.logs_dir, args.exp, log_num, hostname, args.epoch_size)
        # return
    args.log_file = log_file

    # args.img_dir = os.path.expanduser(args.img_dir)
    # img_dir = os.path.join(absolute_base_path, args.img_dir, hostname + '_' + machinename + '_' + args.exp) + '_' + str(log_num)
    # args.img_dir = img_dir

    args.ckpt_dir = os.path.expanduser(args.ckpt_dir)
    ckpt_dir = os.path.join(absolute_base_path, args.ckpt_dir, hostname + '_' + machinename + '_' + args.exp) + '_' + str(log_num) + '_' + args.epoch_size + '_ss'
    args.ckpt_dir = ckpt_dir

    # Create folder to save log files
    os.system('mkdir -p {0}'.format(logs_dir))  # create folder to store log files
    # os.system('mkdir -p {0}'.format(ckpt_dir))  # create folder to store checkpoints

    settings = locals().copy()

    # net.writer = SummaryWriter(comment='_' + args.exp + '_' + args.epoch_size)
    #
    # ckpt_dir = os.path.join(absolute_base_path, net.writer.log_dir)
    # args.ckpt_dir = ckpt_dir

    # Setup output
    def log(text):
        # timestamp = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
        # timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
        timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        # print('[' + timestamp + '] ' + text)
        print(text)
        if log_file is not None:
            with open(log_file, 'a') as f:
                # f.write('[' + timestamp + '] ' + text + '\n')
                str = text + '\n'
                f.write(text + '\n')
                f.flush()
                f.close()
                return str

    cmdline_helpers.ensure_containing_dir_exists(log_file)

    log_str = ''
    log_str += log('\n')
    log_str += log('Output log file {0} created'.format(log_file))
    log_str += log('File used to run the experiment : {0}'.format(absolute_pyfile_path))
    log_str += log('Model files are stored in {0} directory\n'.format(ckpt_dir))

    # Report setttings
    log_str += log(
        'Settings: {}'.format(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))])))

    num_gpu = len(gpus.split(','))
    log_str += log('num_gpu: {0}, GPU-ID: {1}'.format(num_gpu, gpus))

    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
        # if num_gpu < 4:
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

        # set current cuda device to 0
        log_str += log('current cuda device = {}'.format(torch.cuda.current_device()))
        torch.cuda.set_device(0)
        log_str += log('using cuda device = {}'.format(torch.cuda.current_device()))
        batch_size *= int(num_gpu)
        # args.buffer_size = batch_size * 10
        # args.learning_rate *= torch.cuda.device_count()

    else:
        raise EnvironmentError("GPU device not available!")
    try:
        if load_checkpoint == '':
            # define the class centers
            # class0_centroid_radius = 7.5
            # class1_centroid_radius = 2.5
            # domain0_orient_angle = 90  # in degrees
            # domain1_orient_angle = 45  # in degrees
            if args.is_centroid_based:
                class0_centroid_radius = args.class0_centroid_radius
                class1_centroid_radius = args.class1_centroid_radius
                domain0_orient_angle = args.domain0_orient_angle  # in degrees
                domain1_orient_angle = args.domain1_orient_angle  # in degrees
                domain0_orient_angle /= seed * 360.0  # convert to radians
                domain1_orient_angle /= seed * 360.0  # convert to radians

                domain0_class0_centroids = class0_centroid_radius * np.array([np.cos(2*np.pi* domain0_orient_angle), np.sin(2*np.pi* domain0_orient_angle)])
                domain0_class1_centroids = class1_centroid_radius * np.array([np.cos(2*np.pi* domain0_orient_angle), np.sin(2*np.pi* domain0_orient_angle)])
                domain1_class0_centroids = class0_centroid_radius * np.array([np.cos(2*np.pi* domain1_orient_angle), np.sin(2*np.pi* domain1_orient_angle)])
                domain1_class1_centroids = class1_centroid_radius * np.array([np.cos(2*np.pi* domain1_orient_angle), np.sin(2*np.pi* domain1_orient_angle)])

                # Create and plot dataset
                X_src, y_src = make_blobs(n_samples=n_samples, n_features=input_dim, centers=[tuple(domain0_class0_centroids), tuple(domain0_class1_centroids)],
                                          random_state=int(seed * 10), cluster_std=1.0)
                X_tgt, y_tgt = make_blobs(n_samples=n_samples, n_features=input_dim, centers=[tuple(domain1_class0_centroids), tuple(domain1_class1_centroids)],
                                          random_state=int(seed * 20), cluster_std=1.0)
            else:
                # Create and plot dataset
                src_name = 'toy_d0'
                tgt_name = 'toy_d1'
                if not args.swap_domain:
                    X_src, y_src = make_blobs(n_samples=n_samples, n_features=input_dim, centers=2,
                                              random_state=int(seed * 10))
                    X_tgt, y_tgt = make_blobs(n_samples=n_samples, n_features=input_dim, centers=2,
                                              random_state=int(seed * 20))
                    src_name = 'toy_d0'
                    tgt_name = 'toy_d1'
                else:
                    X_tgt, y_tgt = make_blobs(n_samples=n_samples, n_features=input_dim, centers=2,
                                              random_state=int(seed * 10))
                    X_src, y_src = make_blobs(n_samples=n_samples, n_features=input_dim, centers=2,
                                              random_state=int(seed * 20))
                    src_name = 'toy_d1'
                    tgt_name = 'toy_d0'

            # save datasets
            data = \
                {
                    'src': {
                        'X': X_src,
                        'y': y_src,
                    },
                    'tgt': {
                        'X': X_tgt,
                        'y': y_tgt,
                    },
                    'input_dim': input_dim,
                    'n_classes': n_classes,
                    'n_samples': n_samples,
                    'seed': seed,
                    'split_size': split_size,
                    'src_name': src_name,
                    'tgt_name': tgt_name,
            }

        else:
            # load saved datasets
            data = torch.load('{}/{}'.format(absolute_base_path, load_checkpoint))
            # data = \
            #     {
            #         'src': {
            #             'X': d_X_src,
            #             'y': d_y_src,
            #         },
            #         'tgt': {
            #             'X': d_X_tgt,
            #             'y': d_y_tgt,
            #         },
            #         'input_dim': input_dim,
            #         'n_classes': n_classes,
            #         'n_samples': n_samples,
            #         'seed': seed,
            #         'split_size': split_size,
            #     }
            X_src = data['src']['X']
            y_src = data['src']['y']
            X_tgt = data['tgt']['X']
            y_tgt = data['tgt']['y']
            input_dim = data['input_dim']
            n_classes = data['n_classes']
            n_samples = data['n_samples']
            seed = data['seed']
            split_size = data['split_size']
            src_name = data['src_name']
            tgt_name = data['tgt_name']
            log_str += log('Datasets are loaded from {}/{} file.\n'.format(absolute_base_path, load_checkpoint))

        # d_X_src = torch.from_numpy(X_src).float()
        # d_y_src = torch.from_numpy(y_src).long()
        # d_X_tgt = torch.from_numpy(X_tgt).float()
        # d_y_tgt = torch.from_numpy(y_tgt).long()
        # data = \
        #     {
        #         'src': {
        #             'X': d_X_src,
        #             'y': d_y_src,
        #         },
        #         'tgt': {
        #             'X': d_X_tgt,
        #             'y': d_y_tgt,
        #         },
        #         'input_dim': input_dim,
        #         'n_classes': n_classes,
        #         'n_samples': n_samples,
        #         'seed': seed,
        #         'split_size': split_size,
        #     }

        # d_X_src = X_src
        # d_y_src = y_src
        # d_X_tgt = X_tgt
        # d_y_tgt = y_tgt

        from sklearn.model_selection import train_test_split
        X_src_train, X_src_test, y_src_train, y_src_test = train_test_split(X_src, y_src, test_size=1-split_size, random_state=seed)
        X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(X_tgt, y_tgt, test_size=1-split_size, random_state=seed)

        X_src_train = torch.from_numpy(X_src_train).float()
        y_src_train = torch.from_numpy(y_src_train).long()
        X_src_test = torch.from_numpy(X_src_test).float()
        y_src_test = torch.from_numpy(y_src_test).long()
        X_tgt_train = torch.from_numpy(X_tgt_train).float()
        y_tgt_train = torch.from_numpy(y_tgt_train).long()
        X_tgt_test = torch.from_numpy(X_tgt_test).float()
        y_tgt_test = torch.from_numpy(y_tgt_test).long()

        d_src_train = TensorDataset(X_src_train, y_src_train)
        d_src_test = TensorDataset(X_src_test, y_src_test)
        d_tgt_train = TensorDataset(X_tgt_train, y_tgt_train)
        d_tgt_test = TensorDataset(X_tgt_test, y_tgt_test)

        # d_src = TensorDataset(d_X_src, d_y_src)
        # d_tgt = TensorDataset(d_X_tgt, d_y_tgt)
        #
        # train_size = int(split_size * len(d_src))
        # test_size = len(d_src) - train_size
        #
        # d_src_train, d_src_test = dataset.random_split(d_src, [train_size, test_size])
        # d_tgt_train, d_tgt_test = dataset.random_split(d_tgt, [train_size, test_size])

        d_src_train.dataset_name = src_name
        d_tgt_train.dataset_name = tgt_name
        d_src_test.dataset_name = src_name
        d_tgt_test.dataset_name = tgt_name
        d_src_train.n_classes = n_classes
        d_tgt_train.n_classes = n_classes
        d_src_test.n_classes = n_classes
        d_tgt_test.n_classes = n_classes
        X_tgt_noise = Normal(X_tgt_train.mean(dim=0), X_tgt_train.std(dim=0))
        X_src_size = X_tgt_size = torch.Size([batch_size])

        # Grid dataloader for decision boundary
        x_min = min(X_src[:, 0].min(), X_tgt[:, 0].min())
        x_max = max(X_src[:, 0].max(), X_tgt[:, 0].max())
        y_min = min(X_src[:, 1].min(), X_tgt[:, 1].min())
        y_max = max(X_src[:, 1].max(), X_tgt[:, 1].max())
        xx = np.linspace(x_min, x_max, 64)
        yy = np.linspace(y_min, y_max, 64)
        XX, YY = np.meshgrid(xx, yy)
        XX_torch = torch.from_numpy(XX).float().reshape(-1, 1)
        YY_torch = torch.from_numpy(YY).float().reshape(-1, 1)
        X_eval_grid = torch.cat((XX_torch, YY_torch), dim=1)
        d_eval_grid = TensorDataset(X_eval_grid)

        log_str += log('\nSRC : {}: train: count={}, X.shape={} test: count={}, X.shape={}'.format(
            d_src_train.dataset_name.upper(), d_src_train.__len__(), d_src_train[0][0].shape,
            d_src_test.__len__(), d_src_test[0][0].shape))
        log_str += log('TGT : {}: train: count={}, X.shape={} test: count={}, X.shape={}'.format(
            d_tgt_train.dataset_name.upper(), d_tgt_train.__len__(), d_tgt_train[0][0].shape,
            d_tgt_test.__len__(), d_tgt_test[0][0].shape))

        n_samples = max(d_src_train.__len__(), d_tgt_train.__len__())
        if epoch_size == 'large':
            n_samples = max(d_src_train.__len__(), d_tgt_train.__len__())
        elif epoch_size == 'small':
            n_samples = min(d_src_train.__len__(), d_tgt_train.__len__())
        elif epoch_size == 'source':
            n_samples = d_src_train.__len__()
        elif epoch_size == 'target':
            n_samples = d_tgt_train.__len__()
        else:
            raise NotImplementedError

        log_str += log('\nUsing epoch_size : {}'.format(args.epoch_size))

        batch_size = args.batch_size
        n_train_batches = n_samples // batch_size
        n_src_train_batches = d_src_train.__len__() // batch_size
        n_tgt_train_batches = d_tgt_train.__len__() // batch_size
        n_train_samples = n_train_batches * batch_size
        n_src_train_samples = n_src_train_batches * batch_size
        n_tgt_train_samples = n_tgt_train_batches * batch_size

        pin_memory = True
        if use_sampler:
            d_src_train_labels = d_src_train.tensors[1]

            labels, counts_src = np.unique(d_src_train_labels, return_counts=True)

            prior_src = torch.Tensor(counts_src / counts_src.sum()).float().cuda()

            # weights = prior_src.new_tensor(1.0 / counts, dtype=torch.double)
            # weights = torch.DoubleTensor(1.0 / counts)
            counts_src = torch.from_numpy(counts_src).type(torch.double)
            weights_src = 1.0 / counts_src
            sampler_weights_src = weights_src[d_src_train_labels]
            sampler_src = torch.utils.data.sampler.WeightedRandomSampler(weights=sampler_weights_src,
                                                                         num_samples=n_train_samples,
                                                                         replacement=True)
            l_src_train = DataLoader(d_src_train,
                                     batch_size=batch_size,
                                     sampler=sampler_src,
                                     num_workers=int(workers),
                                     pin_memory=pin_memory,
                                     drop_last=True)

            d_tgt_train_labels = d_tgt_train.tensors[1]
            labels, counts_tgt = np.unique(d_tgt_train_labels, return_counts=True)

            prior_tgt = torch.Tensor(counts_tgt / counts_tgt.sum()).float().cuda()

            # weights = prior_tgt.new_tensor(1.0 / counts, dtype=torch.double)
            # weights = torch.DoubleTensor(1.0 / counts)
            counts_tgt = torch.from_numpy(counts_tgt).type(torch.double)
            weights_tgt = 1.0 / counts_tgt
            sampler_weights_tgt = weights_tgt[d_tgt_train_labels]
            sampler_tgt = torch.utils.data.sampler.WeightedRandomSampler(weights=sampler_weights_tgt,
                                                                         num_samples=n_train_samples,
                                                                         replacement=True)
            l_tgt_train = DataLoader(d_tgt_train,
                                     batch_size=batch_size,
                                     sampler=sampler_tgt,
                                     num_workers=int(workers),
                                     pin_memory=pin_memory,
                                     drop_last=True)

            # from sampler import ImbalancedDatasetSampler
            # l_src_train = torch.utils.data.DataLoader(
            #     d_src_train,
            #     sampler=ImbalancedDatasetSampler(d_src_train, num_samples=n_tgt_train_samples),
            #     batch_size=batch_size,
            #     num_workers=int(workers),
            #     drop_last=True)
            #
            # l_tgt_train = torch.utils.data.DataLoader(
            #     d_tgt_train,
            #     sampler=ImbalancedDatasetSampler(d_tgt_train, num_samples=n_tgt_train_samples),
            #     batch_size=batch_size,
            #     num_workers=int(workers),
            #     drop_last=True)

        else:
            shuffle = True
            l_src_train = DataLoader(d_src_train,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=int(workers),
                                     pin_memory=pin_memory,
                                     drop_last=True)

            l_tgt_train = DataLoader(d_tgt_train,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=int(workers),
                                     pin_memory=pin_memory,
                                     drop_last=True)

        all_labels = []
        for i, (X_src, lab) in enumerate(l_src_train):
            all_labels.extend(lab.numpy())
            # os.system('mkdir -p {}'.format(img_dir))
            # image_grid_n_row = int(torch.sqrt(torch.FloatTensor([X_src.shape[0]])).item()) // 2 * 2
            # X_src.data.mul_(0.5).add_(0.5)
            # img_grid = vutils.make_grid(X_src[:image_grid_n_row ** 2, :, :, :],
            #                             nrow=image_grid_n_row)  # make an square grid of images and plot
            # vutils.save_image(img_grid, '{}/{}_train_{:04d}.png'.format(img_dir, d_src_train.dataset_name, i + 1))

        # prior_src_train = np.bincount(all_labels) / len(all_labels)
        # prior_src_train = torch.from_numpy(prior_src_train).float().cuda()
        labels, counts_src = np.unique(all_labels, return_counts=True)

        prior_src_train = torch.Tensor(counts_src / counts_src.sum()).float().cuda()

        log_str += log('prior_src_train : {}'.format(prior_src_train))

        all_labels = []
        for _, lab in l_tgt_train:
            all_labels.extend(lab.numpy())
        # prior_tgt_train = np.bincount(all_labels) / len(all_labels)
        # prior_tgt_train = torch.from_numpy(prior_tgt_train).float().cuda()
        labels, counts_src = np.unique(all_labels, return_counts=True)

        prior_tgt_train = torch.Tensor(counts_src / counts_src.sum()).float().cuda()

        log_str += log('prior_tgt_train : {}'.format(prior_tgt_train))

        # Get Dataloaders from datasets

        shuffle = False
        l_src_test = DataLoader(d_src_test,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=int(2),
                                pin_memory=pin_memory,
                                drop_last=False)

        l_tgt_test = DataLoader(d_tgt_test,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=int(2),
                                pin_memory=pin_memory,
                                drop_last=False)

        l_eval_grid = DataLoader(d_eval_grid,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=int(2),
                                pin_memory=pin_memory,
                                drop_last=False)

        # all_labels = []
        # for _, lab in l_src_test:
        #     all_labels.extend(lab.numpy())
        # prior_src_test = np.bincount(all_labels) / len(all_labels)
        # prior_src_test = torch.from_numpy(prior_src_test).float().cuda()
        # log_str += log('prior_src_test : {}'.format(prior_src_test))
        #
        # all_labels = []
        # for _, lab in l_tgt_test:
        #     all_labels.extend(lab.numpy())
        # prior_tgt_test = np.bincount(all_labels) / len(all_labels)
        # prior_tgt_test = torch.from_numpy(prior_tgt_test).float().cuda()
        # log_str += log('prior_tgt_test : {}'.format(prior_tgt_test))

        net = Network(args)
        # src_train_acc, tgt_train_acc = test(net, d_src_train, d_tgt_train)
        # src_test_acc, tgt_test_acc = test(net, d_src_test, d_tgt_test)
        src_train_acc, tgt_train_acc = test(net, l_src_train, l_tgt_train)
        src_test_acc, tgt_test_acc = test(net, l_src_test, l_tgt_test)

        best_src_test_acc = 0
        best_tgt_test_acc = 0
        if src_test_acc > best_src_test_acc:
            # log_str += log('*** Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
            #     epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
            src_prefix = '  '
            best_src_test_acc = src_test_acc
            net.save_model(suffix='_src_test')
        else:
            # log_str += log('    Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
            #         epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
            src_prefix = '  '

        if tgt_test_acc > best_tgt_test_acc:
            # log_str += log('+++ Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
            #     epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
            tgt_prefix = '++'
            best_tgt_test_acc = tgt_test_acc
            net.save_model(suffix='_tgt_test')

            # tsne = TSNE(n_components=2, verbose=False, perplexity=50, n_iter=600)
            # with torch.no_grad():
            #     for j, (data, target) in enumerate(l_src_test):
            #         if ((j+1)*l_src_test.batch_size) > args.n_test_samples:
            #             break
            #         # data, target = data.to(device), target.to(device)
            #         data, target = data.cuda(), target.cuda()
            #         if net.C.module.use_gumbel:
            #             _, output_logits, output = net.discriminator(data)
            #         else:
            #             _, output_logits = net.discriminator(data)
            #             output = F.softmax(output_logits, dim=1)
            #         src_test_pred_prob, src_test_pred = output.max(1, keepdim=True)  # get the index of the max log-probability
            #
            #         labels_orig = target.cpu().numpy().tolist()
            #         labels_pred = src_test_pred.squeeze().cpu().numpy().tolist()
            #         pred_prob = src_test_pred_prob.squeeze().cpu().numpy().tolist()
            #         if j == 0:
            #             # src_mat = output_logits
            #             src_mat = data
            #             src_pred_mat = output
            #             src_labels_orig = labels_orig
            #             src_labels_pred = labels_pred
            #             src_pred_prob = pred_prob
            #             src_label_img = data
            #         else:
            #             # src_mat = torch.cat((src_mat, output_logits), 0)
            #             src_mat = torch.cat((src_mat, data), 0)
            #             src_pred_mat = torch.cat((src_pred_mat, output), 0)
            #             src_labels_orig.extend(labels_orig)
            #             src_labels_pred.extend(labels_pred)
            #             src_pred_prob.extend(pred_prob)
            #             src_label_img = torch.cat((src_label_img, data), 0)
            #
            #     src_labels_orig_str = list(map(str, src_labels_orig))
            #     src_labels_pred_str = list(map(str, src_labels_pred))
            #     src_metadata = list(map(lambda src_labels_orig_str,
            #                                    src_labels_pred_str: 'SRC_TST_Ori' + src_labels_orig_str + '_Pre' + src_labels_pred_str,
            #                             src_labels_orig_str, src_labels_pred_str))
            #     # label_img.data.mul_(0.5).add_(0.5)
            #     # writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=i)
            #
            #     net.writer.add_pr_curve('SRC_PR', torch.tensor(src_labels_orig), torch.tensor(src_labels_pred), global_step=epoch)
            #
            #     for j, (data, target) in enumerate(l_tgt_test):
            #         if ((j+1)*l_src_test.batch_size) > args.n_test_samples:
            #             break
            #         # data, target = data.to(device), target.to(device)
            #         data, target = data.cuda(), target.cuda()
            #         if net.C.module.use_gumbel:
            #             _, output_logits, output = net.discriminator(data)
            #         else:
            #             _, output_logits = net.discriminator(data)
            #             output = F.softmax(output_logits, dim=1)
            #         tgt_test_pred_pred, tgt_test_pred = output.max(1, keepdim=True)  # get the index of the max log-probability
            #
            #         labels_orig = target.cpu().numpy().tolist()
            #         labels_pred = tgt_test_pred.squeeze().cpu().numpy().tolist()
            #         pred_prob = tgt_test_pred_pred.squeeze().cpu().numpy().tolist()
            #         if j == 0:
            #             # tgt_mat = output_logits
            #             tgt_mat = data
            #             tgt_pred_mat = output
            #             tgt_labels_orig = labels_orig
            #             tgt_labels_pred = labels_pred
            #             tgt_pred_prob = pred_prob
            #             tgt_label_img = data
            #         else:
            #             # tgt_mat = torch.cat((tgt_mat, output_logits), 0)
            #             tgt_mat = torch.cat((tgt_mat, data), 0)
            #             tgt_pred_mat = torch.cat((tgt_pred_mat, output), 0)
            #             tgt_labels_orig.extend(labels_orig)
            #             tgt_labels_pred.extend(labels_pred)
            #             tgt_pred_prob.extend(pred_prob)
            #             tgt_label_img = torch.cat((tgt_label_img, data), 0)
            #
            #     tgt_labels_orig_str = list(map(str, tgt_labels_orig))
            #     tgt_labels_pred_str = list(map(str, tgt_labels_pred))
            #     tgt_metadata = list(map(lambda tgt_labels_orig_str,
            #                                    tgt_labels_pred_str: 'TGT_TST_Ori' + tgt_labels_orig_str + '_Pre' + tgt_labels_pred_str,
            #                             tgt_labels_orig_str, tgt_labels_pred_str))
            #
            #     net.writer.add_pr_curve('TGT_PR', torch.tensor(tgt_labels_orig), torch.tensor(tgt_labels_pred), global_step=epoch)
            #
            #     mat = torch.cat((src_mat, tgt_mat), dim=0)
            #     pred_mat = torch.cat((src_pred_mat, tgt_pred_mat), dim=0)
            #     metadata = src_metadata + tgt_metadata
            #     pred_prob = src_pred_prob + tgt_pred_prob
            #
            #     # label_img = torch.cat((src_label_img, tgt_label_img), dim=0)
            #     # label_img.data.mul_(0.5).add_(0.5)
            #
            #     # net.writer.add_embedding(mat, metadata=metadata, label_img=label_img, n_iter=(epoch * len(tgt_labels_pred)) + j)
            #     net.writer.add_embedding(mat, metadata=metadata, global_step=epoch)
            #
            # # tsne_results = tsne.fit_transform(mat.cpu().numpy())
            # tsne_results = mat.cpu().numpy()
            # pred_mat = pred_mat.cpu().numpy()
            #
            # # # # Plot TSNE
            #
            # fig = plt.figure()
            # # plt.subplot(122)
            # # plt.title("Domain adapted plot", fontsize='small')
            #
            # x, y = tsne_results[:, 0], tsne_results[:, 1]
            # # plt.contour(x, y, np.expand_dims(pred_mat[:, 0], axis=0), levels=0.5, cmap='gray')
            # cax = plt.scatter(tsne_results[:len(src_labels_orig), 0], tsne_results[:len(src_labels_orig), 1],
            #                   c=torch.tensor(src_labels_pred).numpy(), alpha=1.0,
            #                   cmap=plt.cm.get_cmap("jet", n_classes), marker='x',
            #                   label='SRC - {} - #{}'.format(d_src_test.dataset_name.upper(), len(src_labels_orig)))
            # cax = plt.scatter(tsne_results[len(src_labels_orig):, 0], tsne_results[len(src_labels_orig):, 1],
            #                   c=torch.tensor(tgt_labels_pred).numpy(), alpha=1.0,
            #                   cmap=plt.cm.get_cmap("jet", d_tgt_test.n_classes), marker='+',
            #                   label='TGT - {} - #{}'.format(d_tgt_test.dataset_name.upper(), len(tgt_labels_orig)))
            # plt.legend(loc='upper left')
            # # plt.legend(loc='upper right')
            # fig.colorbar(cax, extend='min', ticks=np.arange(0, n_classes))
            # # plt.text(0, 0.1, r'$\delta$',
            # #          {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
            # #           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
            # # plt.figtext(.5, .9, '{} : ACCURACIES\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(args.exp.upper(), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=10)
            # # plt.tight_layout()
            # plt.axis('off')
            # os.system('mkdir -p {}/plots'.format(net.args.ckpt_dir))
            # plt.savefig('{}/plots/{}_tsne_epoch{:03d}.png'.format(net.args.ckpt_dir, net.args.exp, epoch))
            # plt.title('{} : ACCURACIES\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(args.exp.upper(), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=8)
            # plt.savefig('{}/plots/{}_tsne_epoch{:03d}_acc.png'.format(net.args.ckpt_dir, net.args.exp, epoch))
            # # net.writer.add_figure('{}_tsne'.format(args.exp), fig, global_step=(epoch * len(tgt_labels_pred)) + j)
            # plt.tight_layout()
            # net.writer.add_figure('{}_tsne'.format(args.exp), fig, global_step=epoch)

        else:
            # log_str += log('    Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
            #         epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
            tgt_prefix = '  '

        # tsne = TSNE(n_components=2, verbose=False, perplexity=50, n_iter=600)
        with torch.no_grad():
            for j, (data, target) in enumerate(l_src_test):
                if ((j + 1) * l_src_test.batch_size) > args.n_test_samples:
                    break
                # data, target = data.to(device), target.to(device)
                data, target = data.cuda(), target.cuda()
                if net.C.module.use_gumbel:
                    _, output_logits, output = net.discriminator(data)
                else:
                    _, output_logits = net.discriminator(data)
                    output = F.softmax(output_logits, dim=1)
                src_test_pred_prob, src_test_pred = output.max(1,
                                                               keepdim=True)  # get the index of the max log-probability

                labels_orig = target.cpu().numpy().tolist()
                labels_pred = src_test_pred.squeeze().cpu().numpy().tolist()
                pred_prob = src_test_pred_prob.squeeze().cpu().numpy().tolist()
                if j == 0:
                    # src_mat = output_logits
                    src_mat = data
                    src_pred_mat = output
                    src_labels_orig = labels_orig
                    src_labels_pred = labels_pred
                    src_pred_prob = pred_prob
                    src_label_img = data
                else:
                    # src_mat = torch.cat((src_mat, output_logits), 0)
                    src_mat = torch.cat((src_mat, data), 0)
                    src_pred_mat = torch.cat((src_pred_mat, output), 0)
                    src_labels_orig.extend(labels_orig)
                    src_labels_pred.extend(labels_pred)
                    src_pred_prob.extend(pred_prob)
                    src_label_img = torch.cat((src_label_img, data), 0)

            # src_labels_orig_str = list(map(str, src_labels_orig))
            # src_labels_pred_str = list(map(str, src_labels_pred))
            # src_metadata = list(map(lambda src_labels_orig_str,
            #                                src_labels_pred_str: 'SRC_TST_Ori' + src_labels_orig_str + '_Pre' + src_labels_pred_str,
            #                         src_labels_orig_str, src_labels_pred_str))
            # label_img.data.mul_(0.5).add_(0.5)
            # writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=i)

            # net.writer.add_pr_curve('SRC_PR', torch.tensor(src_labels_orig), torch.tensor(src_labels_pred),
            #                         global_step=0)

            for j, (data, target) in enumerate(l_tgt_test):
                if ((j + 1) * l_src_test.batch_size) > args.n_test_samples:
                    break
                # data, target = data.to(device), target.to(device)
                data, target = data.cuda(), target.cuda()
                if net.C.module.use_gumbel:
                    _, output_logits, output = net.discriminator(data)
                else:
                    _, output_logits = net.discriminator(data)
                    output = F.softmax(output_logits, dim=1)
                tgt_test_pred_pred, tgt_test_pred = output.max(1,
                                                               keepdim=True)  # get the index of the max log-probability

                labels_orig = target.cpu().numpy().tolist()
                labels_pred = tgt_test_pred.squeeze().cpu().numpy().tolist()
                pred_prob = tgt_test_pred_pred.squeeze().cpu().numpy().tolist()
                if j == 0:
                    # tgt_mat = output_logits
                    tgt_mat = data
                    tgt_pred_mat = output
                    tgt_labels_orig = labels_orig
                    tgt_labels_pred = labels_pred
                    tgt_pred_prob = pred_prob
                    tgt_label_img = data
                else:
                    # tgt_mat = torch.cat((tgt_mat, output_logits), 0)
                    tgt_mat = torch.cat((tgt_mat, data), 0)
                    tgt_pred_mat = torch.cat((tgt_pred_mat, output), 0)
                    tgt_labels_orig.extend(labels_orig)
                    tgt_labels_pred.extend(labels_pred)
                    tgt_pred_prob.extend(pred_prob)
                    tgt_label_img = torch.cat((tgt_label_img, data), 0)

            # tgt_labels_orig_str = list(map(str, tgt_labels_orig))
            # tgt_labels_pred_str = list(map(str, tgt_labels_pred))
            # tgt_metadata = list(map(lambda tgt_labels_orig_str,
            #                                tgt_labels_pred_str: 'TGT_TST_Ori' + tgt_labels_orig_str + '_Pre' + tgt_labels_pred_str,
            #                         tgt_labels_orig_str, tgt_labels_pred_str))

            # net.writer.add_pr_curve('TGT_PR', torch.tensor(tgt_labels_orig), torch.tensor(tgt_labels_pred),
            #                         global_step=0)

            mat = torch.cat((src_mat, tgt_mat), dim=0)
            pred_mat = torch.cat((src_pred_mat, tgt_pred_mat), dim=0)
            # metadata = src_metadata + tgt_metadata
            pred_prob = src_pred_prob + tgt_pred_prob

            # label_img = torch.cat((src_label_img, tgt_label_img), dim=0)
            # label_img.data.mul_(0.5).add_(0.5)

            # net.writer.add_embedding(mat, metadata=metadata, label_img=label_img, n_iter=(epoch * len(tgt_labels_pred)) + j)
            # net.writer.add_embedding(mat, metadata=metadata, global_step=0)

        # tsne_results = tsne.fit_transform(mat.cpu().numpy())
        tsne_results = mat.cpu().numpy()
        pred_mat = pred_mat.cpu().numpy()

        # # # Plot TSNE

        fig = plt.figure()
        # plt.subplot(122)
        # plt.title("Domain adapted plot", fontsize='small')

        x, y = tsne_results[:, 0], tsne_results[:, 1]
        # plt.contour(x, y, np.expand_dims(pred_mat[:, 0], axis=0), levels=0.5, cmap='gray')
        cax = plt.scatter(tsne_results[:len(src_labels_orig), 0], tsne_results[:len(src_labels_orig), 1],
                          c=torch.tensor(src_labels_orig).numpy(), alpha=1.0,
                          cmap=plt.cm.get_cmap("jet", n_classes), marker='x',
                          label='SRC - {} - #{}'.format(d_src_test.dataset_name.upper(), len(src_labels_orig)))
        cax = plt.scatter(tsne_results[len(src_labels_orig):, 0], tsne_results[len(src_labels_orig):, 1],
                          c=torch.tensor(tgt_labels_orig).numpy(), alpha=1.0,
                          cmap=plt.cm.get_cmap("jet", d_tgt_test.n_classes), marker='+',
                          label='TGT - {} - #{}'.format(d_tgt_test.dataset_name.upper(), len(tgt_labels_orig)))
        plt.legend(loc='upper left')
        # plt.legend(loc='upper right')
        fig.colorbar(cax, extend='min', ticks=np.arange(0, n_classes))
        # plt.text(0, 0.1, r'$\delta$',
        #          {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
        #           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
        # plt.figtext(.5, .9, '{} : ACCURACIES\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(args.exp.upper(), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=10)
        # plt.tight_layout()
        plt.axis('off')
        os.system('mkdir -p {}/plots'.format(net.args.ckpt_dir))
        os.system('mkdir -p {}/plots/tsne'.format(net.args.ckpt_dir))
        os.system('mkdir -p {}/plots/tsne_acc'.format(net.args.ckpt_dir))
        os.system('mkdir -p {}/plots/contour'.format(net.args.ckpt_dir))
        os.system('mkdir -p {}/plots/contour_acc'.format(net.args.ckpt_dir))
        plt.savefig('{}/plots/src_{}_seed_{:04d}_ss_original.png'.format(net.args.ckpt_dir, src_name, seed))
        plt.title('{} : ACCURACIES : EPOCH -1\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(
            args.exp.upper(), src_train_acc / (l_src_train.__len__() * batch_size),
                              tgt_train_acc / (l_tgt_train.__len__() * batch_size), src_test_acc / len(d_src_test),
                              tgt_test_acc / len(d_tgt_test)), fontsize=8)
        plt.savefig('{}/plots/src_{}_seed_{:04d}_ss_original_acc.png'.format(net.args.ckpt_dir, src_name, seed))
        # net.writer.add_figure('{}_tsne'.format(args.exp), fig, global_step=(epoch * len(tgt_labels_pred)) + j)
        plt.tight_layout()
        net.writer.add_figure('src_{}_seed_{:04d}_ss_original_acc'.format(src_name, seed), fig, global_step=0)

        # log_str += log('{}{} Epoch:{:03d}/{:03d} #batch:{:03d} took {:03.2f}m - Loss:{:03.5f}, SRC_TRAIN_ACC = {:2.2%}, TGT_TRAIN_ACC = {:2.2%}, SRC_TEST_ACC = {:2.2%}, TGT_TEST_ACC = {:2.2%}'.format(
        #     src_prefix, tgt_prefix, epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
        log(
            '{}{} E:{:03d}/{:03d} #B:{:03d}, t={:06.2f}m, L={:07.4f}, ACC : S_TRN= {:5.2%}, T_TRN= {:5.2%}, S_TST= {:5.2%}, T_TST= {:5.2%}'.format(
                src_prefix, tgt_prefix, 0, num_epochs, i + 1, 0.0, 0.0 / (i + 1),
                                                           src_train_acc / (l_src_train.__len__() * batch_size),
                                                           tgt_train_acc / (l_tgt_train.__len__() * batch_size),
                                                           src_test_acc / len(d_src_test),
                                                           tgt_test_acc / len(d_tgt_test)))

        net.writer.add_scalar('Accuracy/train/src', src_train_acc / (l_src_train.__len__() * batch_size), 0)
        net.writer.add_scalar('Accuracy/train/tgt', tgt_train_acc / (l_tgt_train.__len__() * batch_size), 0)
        net.writer.add_scalar('Accuracy/test/src', src_test_acc / len(d_src_test), 0)
        net.writer.add_scalar('Accuracy/test/tgt', tgt_test_acc / len(d_tgt_test), 0)

        net.writer.add_text('epoch log',
                            '{}{} E:{:03d}/{:03d} #B:{:03d}, t={:06.2f}m, L={:07.4f}, ACC : S_TRN= {:5.2%}, T_TRN= {:5.2%}, S_TST= {:5.2%}, T_TST= {:5.2%}'.format(
                                src_prefix, tgt_prefix, 0, num_epochs, i + 1, 0.0, 0.0 / (i + 1),
                                                                           src_train_acc / (
                                                                                       l_src_train.__len__() * batch_size),
                                                                           tgt_train_acc / (
                                                                                       l_tgt_train.__len__() * batch_size),
                                                                           src_test_acc / len(d_src_test),
                                                                           tgt_test_acc / len(d_tgt_test)), 0)
        os.system('cp {0} {1}/Log/'.format(log_file, net.args.ckpt_dir))

        # Save the current dataset
        torch.save(data, '{}/{}.pt'.format(ckpt_dir, exp))
        log_str += log('Dataset files are stored in {0} directory\n'.format(ckpt_dir))

        # fig = plt.figure()
        # plt.title("Original dataset", fontsize='small')
        # cax = plt.scatter(X_src_test[:, 0], X_src_test[:, 1],
        #                   c=y_src_test, alpha=1.0,
        #                   cmap=plt.cm.get_cmap("jet", n_classes), marker='x',
        #                   label='SRC - #{}'.format(len(y_src)))
        # cax = plt.scatter(X_tgt_test[:, 0], X_tgt_test[:, 1],
        #                   c=y_tgt_test, alpha=1.0,
        #                   cmap=plt.cm.get_cmap("jet", n_classes), marker='+',
        #                   label='TGT - #{}'.format(len(y_tgt)))
        # plt.legend(loc='upper left')
        # # plt.legend(loc='upper right')
        # fig.colorbar(cax, extend='min', ticks=np.arange(0, n_classes))
        # # plt.text(0, 0.1, r'$\delta$',
        # #          {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
        # #           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
        # # plt.figtext(.5, .9, '{} : ACCURACIES\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(exp.upper(), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=10)
        # plt.tight_layout()
        # plt.axis('off')
        # os.system('mkdir -p {}/plots'.format(ckpt_dir))
        # plt.savefig('{}/plots/01{}_original_dataset.png'.format(ckpt_dir, exp))

        # plt.show()

        log_str += log('\nBuilding Network from {} ...'.format(args.network_type.upper()))
        log_str += log('Encoder : {}'.format(net.E))
        log_str += log('Classifier : {}'.format(net.C))
        log_str += log('Network Built ...')
        
        # For storing the network output of the last buffer_size samples
        p_sum_denominator = torch.rand(args.buffer_size, n_classes).cuda()
        p_sum_denominator /= p_sum_denominator.sum(1).unsqueeze(1).expand_as(p_sum_denominator)
        # p_sum_denominator_src = torch.rand(args.buffer_size, n_classes).cuda()
        # # p_sum_denominator_src = torch.rand(n_tgt_train_samples, n_classes).cuda()
        # p_sum_denominator_src /= p_sum_denominator_src.sum(1).unsqueeze(1).expand_as(p_sum_denominator_src)
        # p_sum_denominator_tgt = p_sum_denominator_src.clone().detach()

        # setup optimizer
        log_str += log('\noptimizerE : {}'.format(net.optimizerE))
        log_str += log('optimizerC : {}'.format(net.optimizerC))

        # Loss function for supervised loss
        classification_criterion = nn.CrossEntropyLoss().cuda()

        # discriminator_parameters = filter(lambda p: p.requires_grad, discriminator.parameters())
        # params = sum([np.prod(p.size()) for p in discriminator_parameters])
        #
        # log_str += log('Number of Trainable Parameters: ' + str(params))

        log_str += log('\nTraining...')
        if simul_train_src_tgt:
            log_str += log('Note : Simultaneous training of source and target domains. No swapping after e epochs ...')
        else:
            log_str += log('Note : No Simultaneous training of source and target domains. swapping after e epochs ...')

        X_tgt_noise = Normal(X_tgt_train.mean(dim=0), X_tgt_train.std(dim=0))
        best_src_test_acc = 0  # Best epoch wise src acc
        best_src_test_acc_inter = 0  # Best generator wise src acc
        best_tgt_test_acc = 0  # Best epoch wise src acc
        best_tgt_test_acc_inter = 0  # Best generator wise src acc
        src_test_acc = 0  # dummy init for scheduler

        time = timeit.default_timer()
        total_loss_epochs = []

        src_log_prob_buffer = []  # src_log_prob_buffer
        tgt_log_prob_buffer = []  # tgt_log_prob_buffer
        z_src_one_hot_buffer = []  # src_log_prob_buffer
        z_tgt_one_hot_buffer = []  # tgt_log_prob_buffer

        log_str += log('Checkpoint directory to store files for current run : {}'.format(net.args.ckpt_dir))

        net.writer.add_text('Log Text', log_str)

        # # Reset dataloader iterator
        # src_data_iter = iter(l_src_train)
        # tgt_data_iter = iter(l_tgt_train)
        epoch = -1
        counter = 0
        for epoch in range(num_epochs):
        # while True:
        #     epoch += 1
        #     if tgt_test_acc/len(d_tgt_test) >= 0.999 or tgt_test_acc/len(d_tgt_test) <= 0.001 :
        #         counter +=1
        #         if counter > 20:
        #             break  # stop training after 100

            if epoch != 0 and args.lr_decay_type == 'scheduler':
                # scheduler_E.step(src_test_acc)
                # scheduler_C.step(src_test_acc)
                # scheduler_GS.step(src_test_acc)
                # scheduler_GT.step(src_test_acc)
                lr = args.learning_rate
                pass
            elif args.lr_decay_type == 'geometric':
                lr = args.learning_rate * (args.lr_decay_rate ** (epoch // args.lr_decay_period))
                for param_group in net.optimizerC.param_groups:
                    param_group['lr'] = lr
                for param_group in net.optimizerE.param_groups:
                    param_group['lr'] = lr
            else:
                lr = args.learning_rate

            if use_ramp_sup:
                epoch_ramp_2 = epoch % (args.ramp * 2)
                ramp_sup_value = 1.0
                # if epoch_ramp_2 < (args.ramp):
                #     ramp_sup_value = 1.0 - math.exp(-(args.ramp - epoch_ramp_2) * 5.0 / args.ramp)
                # else:
                #     ramp_sup_value = math.exp(-(args.ramp * 2 - epoch_ramp_2) * 5.0 / args.ramp)

                ramp_sup_weight_in_list[0] = ramp_sup_value

            if use_ramp_unsup:
                epoch_ramp_2 = epoch % (args.ramp * 2)
                if epoch < (args.ramp):
                    # p = max(0.0, float(epoch)) / float(args.ramp)
                    # p = 1.0 - p
                    ramp_unsup_value = math.exp(-(args.ramp - epoch) * 5.0 / args.ramp)
                else:
                    # ramp_unsup_value = 1.0
                    ramp_unsup_value = 1.0 - math.exp(-(args.ramp - epoch_ramp_2) * 5.0 / args.ramp)

                ramp_unsup_weight_in_list[0] = ramp_unsup_value

            net.writer.add_scalar('constant/learning_rate', lr, epoch)
            net.writer.add_scalar('weights_loss/sup/src', lambda_ssl, epoch)
            # net.writer.add_scalar('weights_loss/unsup/src', lambda_sul, epoch)
            # net.writer.add_scalar('weights_loss/unsup/tgt', lambda_tul, epoch)
            # net.writer.add_scalar('weights_loss/adv/src', lambda_sal, epoch)
            # net.writer.add_scalar('weights_loss/adv/tgt', lambda_tal, epoch)
            # net.writer.add_scalar('weights_loss/sup/src_ramp', ramp_sup_weight_in_list[0], epoch)
            # net.writer.add_scalar('weights_loss/unsup/src_ramp', ramp_unsup_weight_in_list[0], epoch)
            # net.writer.add_scalar('weights_loss/unsup/tgt_ramp', ramp_unsup_weight_in_list[0], epoch)

            net.E.train()
            net.C.train()
            epoch_loss = 0
            # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

            # # Reset dataloader iterator
            # src_data_iter = iter(l_src_train)
            # tgt_data_iter = iter(l_tgt_train)
            # y_src_all = []

            # for i, ((X_src, y_src), (X_tgt, _)) in enumerate(zip(cycle(l_src_train), l_tgt_train)):
            # if use_sampler:
            for i, ((X_src, y_src), (X_tgt, _)) in enumerate(zip(l_src_train, l_tgt_train)):
                n_iter = (epoch * len(l_src_train)) + i
                net.zero_grad()

                # y_src_all.extend(y_src.numpy())

                # try:  # Incase smaller dataset iter is over, Start new
                #     X_src, y_src = src_data_iter.next()  # Only Source Domain Labels should be used
                # except:
                #     src_data_iter = iter(l_src_train)
                #     X_src, y_src = src_data_iter.next()  # Only Source Domain Labels should be used
                #
                # try:  # Incase smaller dataset iter is over, Start new
                #     X_tgt, _ = tgt_data_iter.next()  # Target Domain Labels should not be used
                # except:
                #     tgt_data_iter = iter(l_tgt_train)
                #     X_tgt, _ = tgt_data_iter.next()  # Target Domain Labels should not be used
            # # if Not use_sampler:
            # for i in range(n_train_batches):
            #     net.zero_grad()
            #
            #     try:  # Incase smaller dataset iter is over, Start new
            #         X_src, y_src = src_data_iter.next()  # Only Source Domain Labels should be used
            #     except:
            #         src_data_iter = iter(l_src_train)
            #         X_src, y_src = src_data_iter.next()  # Only Source Domain Labels should be used
            #
            #     try:  # Incase smaller dataset iter is over, Start new
            #         X_tgt, _ = tgt_data_iter.next()  # Target Domain Labels should not be used
            #     except:
            #         tgt_data_iter = iter(l_tgt_train)
            #         X_tgt, _ = tgt_data_iter.next()  # Target Domain Labels should not be used

                total_loss = 0

                # Get Source and Target Images
                X_src = Variable(X_src.cuda())
                # X_tgt = Variable(X_tgt.cuda())
                X_src_batch_size = X_src.size(0)
                # X_tgt_batch_size = X_tgt.size(0)

                # Exit without processing last uneven batch
                # if X_src_batch_size != X_tgt_batch_size:
                #     break

                # Get only Source labels
                y_src = Variable(y_src.cuda())

                # Train Enc + Dec + Classifier on both Source and Target Domain data
                if net.C.module.use_gumbel:
                    src_enc_out, src_logits_out, z_src_one_hot = net.discriminator(X_src)
                    # tgt_enc_out, tgt_logits_out, z_tgt_one_hot = net.discriminator(X_tgt)
                else:
                    src_enc_out, src_logits_out = net.discriminator(X_src)
                    # tgt_enc_out, tgt_logits_out = net.discriminator(X_tgt)

                # Supervised classification loss
                src_sup_loss = classification_criterion(src_logits_out, y_src)  # Loss 1 : Supervised loss  # torch.Size([64, 10])

                # # # Train Enc + Dec + Classifier on both Source and Target Domain data
                # # if net.C.module.use_gumbel:
                # #     src_enc_out, src_logits_out, z_src_one_hot = net.discriminator(X_src)
                # #     tgt_enc_out, tgt_logits_out, z_tgt_one_hot = net.discriminator(X_tgt)
                # # else:
                # #     src_enc_out, src_logits_out = net.discriminator(X_src)
                # #     tgt_enc_out, tgt_logits_out = net.discriminator(X_tgt)
                #
                # # Calculating unsupervised loss and and log loss
                # src_prob = F.softmax(src_logits_out, dim=1)  # torch.Size([64, 10])
                # src_log_prob = F.log_softmax(src_logits_out, dim=1)  # torch.Size([64, 10])
                # tgt_prob = F.softmax(tgt_logits_out, dim=1)  # torch.Size([64, 10])
                # tgt_log_prob = F.log_softmax(tgt_logits_out, dim=1)  # torch.Size([64, 10])

                # if not net.C.module.use_gumbel:
                #     with torch.no_grad():
                #         # expectation step
                #         # src_prob_cp = src_prob.data
                #         # tgt_prob_cp = tgt_prob.data
                #         # src_prob_cp = src_prob.detach().clone()
                #         tgt_prob_cp = tgt_prob.detach().clone()
                #
                #         # Update denominator for the unsupervised loss
                #         p_sum_denominator = torch.cat((tgt_prob_cp, p_sum_denominator), 0)[0:args.buffer_size]
                #         # p_sum_denominator = torch.cat((src_prob_cp, tgt_prob_cp, p_sum_denominator), 0)[0:args.buffer_size]
                #         # src_prob_cp *= (prior_src_train / p_sum_denominator.sum(0)).expand_as(src_prob_cp)
                #         tgt_prob_cp *= (prior_src_train / p_sum_denominator.sum(0)).expand_as(tgt_prob_cp)
                #         # p_sum_denominator_src = torch.cat((src_prob_cp, p_sum_denominator_src), 0)[0:args.buffer_size]
                #         # src_prob_cp *= (prior_src / p_sum_denominator_src.sum(0)).expand_as(src_prob_cp)
                #         # p_sum_denominator_tgt = torch.cat((tgt_prob_cp, p_sum_denominator_tgt), 0)[0:args.buffer_size]
                #         # tgt_prob_cp *= (prior_src / p_sum_denominator_tgt.sum(0)).expand_as(tgt_prob_cp)
                #         # p_sum_denominator_src = torch.cat((src_prob_cp, p_sum_denominator_src), 0)[0:n_tgt_train_samples]
                #         # src_prob_cp *= (prior_src / p_sum_denominator_src.sum(0)).expand_as(src_prob_cp)
                #         # p_sum_denominator_tgt = torch.cat((tgt_prob_cp, p_sum_denominator_tgt), 0)[0:n_tgt_train_samples]
                #         # tgt_prob_cp *= (prior_src / p_sum_denominator_tgt.sum(0)).expand_as(tgt_prob_cp)
                #
                #         # Find expected labels
                #         # _, y_src_pred = src_prob_cp.max(dim=1)
                #         # z_src_one_hot = torch.FloatTensor(y_src_pred.size(0), n_classes).cuda()
                #         # z_src_one_hot.zero_()
                #         # z_src_one_hot.scatter_(1, y_src_pred.unsqueeze(1), 1)
                #         # z_src_one_hot = Variable(z_src_one_hot)
                #
                #         _, y_tgt_pred = tgt_prob_cp.max(dim=1)
                #         z_tgt_one_hot = torch.FloatTensor(X_tgt_batch_size, n_classes).cuda()
                #         z_tgt_one_hot.zero_()
                #         z_tgt_one_hot.scatter_(1, y_tgt_pred.unsqueeze(1), 1)
                #         z_tgt_one_hot = Variable(z_tgt_one_hot)
                #
                # # maximization step
                #
                # # src_exponent = torch.mm(z_src_one_hot, src_log_prob.t())
                # # src_exponent_new = src_exponent - torch.diag(src_exponent).view(X_src_batch_size, 1).expand_as(src_exponent)
                # #
                # # # src_temp = src_exponent_new.exp()
                # # # src_p_x_z_inv = src_temp.sum(dim=1)
                # # # src_unsup_loss = src_p_x_z_inv.log().mean()
                # # src_unsup_loss = torch.logsumexp(src_exponent_new, dim=1).mean()
                #
                # # maximization step
                # tgt_exponent = torch.mm(z_tgt_one_hot, tgt_log_prob.t())
                # tgt_exponent_new = tgt_exponent - torch.diag(tgt_exponent).view(X_tgt_batch_size, 1).expand_as(tgt_exponent)
                #
                # # tgt_temp = tgt_exponent_new.exp()
                # # tgt_p_x_z_inv = tgt_temp.sum(dim=1)
                # # tgt_unsup_loss = tgt_p_x_z_inv.log().mean()
                # tgt_unsup_loss = torch.logsumexp(tgt_exponent_new, dim=1).mean()

                # src_unsup_loss = F.log_softmax(torch.mm(z_src_one_hot, src_log_prob.t()).diag(), dim=0).mean() * net.mone  # torch.Size([64, 10])
                # tgt_unsup_loss = F.log_softmax(torch.mm(z_tgt_one_hot, tgt_log_prob.t()).diag(), dim=0).mean() * net.mone  # torch.Size([64, 10])

                # # # Adversarial regularization loss
                # with torch.no_grad():
                # #     # X_src_noise = torch.FloatTensor(X_src_batch_size, gen_in_dim, 1, 1).normal_(0, 1).cuda()
                # #     # X_tgt_noise = torch.FloatTensor(X_tgt_batch_size, gen_in_dim, 1, 1).normal_(0, 1).cuda()
                # #     X_src_noise = Variable(X_src_noise.normal_(0, 1))  # total freeze netG_src , netG_tgt
                # #     X_tgt_noise = Variable(X_tgt_noise.normal_(0, 1))  # total freeze netG_src , netG_tgt
                # #     # X_src_noise.normal_(0, 1)  # total freeze netG_src , netG_tgt
                # #     # X_tgt_noise.normal_(0, 1)  # total freeze netG_src , netG_tgt
                # #     # X_src_gen = net.GS(X_src_noise).detach()
                # #     # X_tgt_gen = net.GT(X_tgt_noise).detach()
                # #     X_src_gen = net.GS(X_src_noise)
                # #     X_tgt_gen = net.GT(X_tgt_noise)
                #       X_tgt_gen = X_tgt_noise.sample(X_tgt_size).cuda()
                # #
                # if net.C.module.use_gumbel:
                # #     src_gen_enc_out, src_gen_logits_out, _ = net.discriminator(X_src_gen)
                #       tgt_gen_enc_out, tgt_gen_logits_out, _ = net.discriminator(X_tgt_gen)
                # else:
                # #     src_gen_enc_out, src_gen_logits_out = net.discriminator(X_src_gen)
                #       tgt_gen_enc_out, tgt_gen_logits_out = net.discriminator(X_tgt_gen)
                # #
                # # src_adv_loss = F.log_softmax(src_gen_logits_out, dim=1).mean(dim=1).mean() * net.mone
                # tgt_adv_loss = F.log_softmax(tgt_gen_logits_out, dim=1).mean(dim=1).mean() * net.mone

                # lambda_ssl = ramp_sup_weight_in_list[0]
                # lambda_sul = lambda_tul = ramp_unsup_weight_in_list[0]
                total_classifier_loss = lambda_ssl * ramp_sup_weight_in_list[0] * src_sup_loss # + lambda_tul * ramp_unsup_weight_in_list[0] * tgt_unsup_loss # + lambda_sul * ramp_unsup_weight_in_list[0] * src_unsup_loss + lambda_tal * tgt_adv_loss  # Total Encoder + Classifier loss for Enc + Class training

                ### Add loss and loss weights to tensorboard logs
                net.writer.add_scalar('classifier_loss/sup/src', src_sup_loss.item(), n_iter)
                # net.writer.add_scalar('classifier_loss/unsup/src', src_unsup_loss.item(), n_iter)
                # net.writer.add_scalar('classifier_loss/unsup/tgt', tgt_unsup_loss.item(), n_iter)
                # net.writer.add_scalar('classifier_loss/adv/src', src_adv_loss.item(), n_iter)
                # net.writer.add_scalar('classifier_loss/adv/tgt', tgt_adv_loss.item(), n_iter)
                net.writer.add_scalar('classifier_loss/total/batch', total_classifier_loss.item(), n_iter)

                total_loss += total_classifier_loss

                total_loss.backward()
                net.optimizerE.step()
                net.optimizerC.step()

                epoch_loss += total_loss.item()
                run_time = (timeit.default_timer() - time) / 60.0

                # log(
                #     '    [%3d/%3d][%4d/%4d] (%.2f m): ttl=%.6f tcl=%.6f ssl=%.6f sul=%.6f tul=%.6f errd=%.6f mmdl=%.6f ose=%.6f srl=%.6f trl=%.6f'
                #     % (epoch, num_epochs, i, n_train_batches, run_time,
                #        total_loss.item(), total_classifier_loss.data[0], src_sup_loss.data[0],
                #        src_unsup_loss.data[0], tgt_unsup_loss.data[0],
                #        errD.data[0], mmd2_D.data[0] * -1, one_side_errD.data[0] * -1,
                #        L2_AE_X_src_D.data[0], L2_AE_X_tgt_D.data[0]))

                # ### generator traininig starts
                # net.zero_grad()
                #
                # # X_src_noise = torch.FloatTensor(X_src_batch_size, gen_in_dim, 1, 1).normal_(0, 1).cuda()
                # # X_tgt_noise = torch.FloatTensor(X_tgt_batch_size, gen_in_dim, 1, 1).normal_(0, 1).cuda()
                # # X_src_noise = Variable(X_src_noise)  # total freeze netG_src , netG_tgt
                # # X_tgt_noise = Variable(X_tgt_noise)  # total freeze netG_src , netG_tgt
                # X_src_noise.normal_(0, 1)  # total freeze netG_src , netG_tgt
                # X_tgt_noise.normal_(0, 1)  # total freeze netG_src , netG_tgt
                # X_src_gen = net.GS(X_src_noise)
                # X_tgt_gen = net.GT(X_tgt_noise)
                #
                # if net.C.module.use_gumbel:
                #     src_enc_out, src_logits_out, _ = net.discriminator(X_src)
                #     src_gen_enc_out, src_gen_logits_out, _ = net.discriminator(X_src_gen)
                #     tgt_enc_out, tgt_logits_out, _ = net.discriminator(X_tgt)
                #     tgt_gen_enc_out, tgt_gen_logits_out, _ = net.discriminator(X_tgt_gen)
                # else:
                #     src_enc_out, src_logits_out = net.discriminator(X_src)
                #     src_gen_enc_out, src_gen_logits_out = net.discriminator(X_src_gen)
                #     tgt_enc_out, tgt_logits_out = net.discriminator(X_tgt)
                #     tgt_gen_enc_out, tgt_gen_logits_out = net.discriminator(X_tgt_gen)
                #
                # src_m1 = src_enc_out.mean(0)
                # src_m2 = src_gen_enc_out.mean(0)
                # tgt_m1 = tgt_enc_out.mean(0)
                # tgt_m2 = tgt_gen_enc_out.mean(0)
                #
                # src_mmd2_D = mix_rbf_mmd2(src_enc_out, src_gen_enc_out, sigma_list)
                # src_mmd2_D = F.relu(src_mmd2_D)
                # tgt_mmd2_D = mix_rbf_mmd2(tgt_enc_out, tgt_gen_enc_out, sigma_list)
                # tgt_mmd2_D = F.relu(tgt_mmd2_D)
                #
                # if use_gen_sqrt:
                #     loss_src_gen = (src_m1 - src_m2).abs().mean() + torch.sqrt(src_mmd2_D)
                #     loss_tgt_gen = (tgt_m1 - tgt_m2).abs().mean() + torch.sqrt(tgt_mmd2_D)
                # else:
                #     loss_src_gen = (src_m1-src_m2).abs().mean() + src_mmd2_D
                #     loss_tgt_gen = (tgt_m1-tgt_m2).abs().mean() + tgt_mmd2_D
                #
                # loss_gen = loss_src_gen + loss_tgt_gen
                #
                # loss_gen.backward()
                #
                # net.writer.add_scalar('generator_loss/adv/src', loss_src_gen.item(), n_iter)
                # net.writer.add_scalar('generator_loss/adv/tgt', loss_tgt_gen.item(), n_iter)
                # net.writer.add_scalar('generator_loss/total/batch', loss_gen.item(), n_iter)
                #
                # net.optimizerGS.step()
                # if not args.use_tied_gen:
                #     net.optimizerGT.step()
                #
                # if train_GnE:  # use_gen_sqrt, train_GnE
                #     net.optimizerE.step()
                #
                # # Generator training ends
                # if n_iter % plot_interval == 0:
                #     # log(
                #     #     '    [%3d/%3d][%4d/%4d] (%.2f m): ttl=%.6f tcl=%.6f ssl=%.6f sul=%.6f tul=%.6f sal=%.6f tal=%.6f'
                #     #     % (epoch, num_epochs, i, n_train_batches, run_time,
                #     #        total_loss.item(), total_classifier_loss.item(), src_sup_loss.item(),
                #     #        src_unsup_loss.item(), tgt_unsup_loss.item(),
                #     #        src_adv_loss.item(), tgt_adv_loss.item()))
                #
                #     # Save image using torch vision
                #     with torch.no_grad():
                #         fixed_noise = Variable(fixed_noise)  # total freeze netG_src , netG_tgt
                #         X_fixed_noise_S = net.GS(fixed_noise).detach()
                #         X_fixed_noise_T = net.GT(fixed_noise).detach()
                #
                #         image_grid = torch.zeros(
                #             (X_fixed_noise_S.shape[0] * 2, X_fixed_noise_S.shape[1], X_fixed_noise_S.shape[2],
                #              X_fixed_noise_S.shape[3]),
                #             requires_grad=False)
                #         # image_grid[0::4, :, :, :] = X_src
                #         # image_grid[2::4, :, :, :] = X_tgt
                #         image_grid[0::2, :, :, :] = X_fixed_noise_S
                #         image_grid[1::2, :, :, :] = X_fixed_noise_T
                #         # image_grid = image_grid * std + mean
                #         image_grid_n_row = int(torch.sqrt(torch.FloatTensor([X_fixed_noise_S.shape[0]])).item())//2*2
                #         image_grid.data.mul_(0.5).add_(0.5)
                #         img_grid = vutils.make_grid(image_grid[:image_grid_n_row ** 2, :, :, :], nrow=image_grid_n_row)  # make an square grid of images and plot
                #
                #         net.writer.add_image('Generator_images_{0}'.format(args.exp), img_grid, n_iter)  # Tensor
                #         # vutils.save_image(img_grid, '{}/{}_train_{:04d}_{:04d}.png'.format(img_dir, args.dataset, epoch + 1, batch_idx + 1))



            # Show the loss
            # log_str += log('Epoch:{:04d}\tLoss:{:6.5f}\n'.format(epoch + 1,  epoch_loss/i))
            total_loss_epochs.append(epoch_loss/(i+1))
            net.writer.add_scalar('classifier_loss/total/epoch', epoch_loss/(i+1), epoch)

            # Create folder to save images
            if epoch == 0:
                # os.system('mkdir -p {0}/Image'.format(args.ckpt_dir))  # create folder to store images
                os.system('mkdir -p {0}/Log'.format(net.args.ckpt_dir))  # create folder to store images

                # vutils.save_image(img_grid, '{}/Image/{}_train_e{:03d}.png'.format(args.ckpt_dir, args.exp, epoch + 1))

            # if epoch % 30 == 0:
            #     # p_sum_denominator.rand_()
            #     # p_sum_denominator /= p_sum_denominator.sum(1).unsqueeze(1).expand_as(p_sum_denominator)
            #     # Create folder to save model checkpoints
            #     net.save_model(suffix='_bkp')
            #     # os.system('mkdir -p {0}/Log'.format(ckpt_dir))  # create folder to store logs
            #     image_grid = torch.zeros(
            #         (X_src.shape[0] * 2, X_src.shape[1], X_src.shape[2],
            #          X_src.shape[3]),
            #         requires_grad=False)
            #     # image_grid[0::4, :, :, :] = X_src
            #     # image_grid[2::4, :, :, :] = X_tgt
            #     image_grid[0::2, :, :, :] = X_src
            #     image_grid[1::2, :, :, :] = X_tgt
            #     # image_grid = image_grid * std + mean
            #     image_grid_n_row = int(torch.sqrt(torch.FloatTensor([X_src.shape[0]])).item()) // 2 * 2
            #     image_grid.data.mul_(0.5).add_(0.5)
            #     img_grid = vutils.make_grid(image_grid[:image_grid_n_row ** 2, :, :, :],
            #                                 nrow=image_grid_n_row)  # make an square grid of images and plot
            #
            #     net.writer.add_image('Original_images_{0}'.format(args.exp), img_grid, epoch)  # Tensor
            #     # vutils.save_image(img_grid, '{}/{}_train_{:04d}_{:04d}.png'.format(img_dir, args.dataset, epoch + 1, batch_idx + 1))

            # src_train_acc, tgt_train_acc = test(net, d_src_train, d_tgt_train)
            # src_test_acc, tgt_test_acc = test(net, d_src_test, d_tgt_test)
            src_train_acc, tgt_train_acc = test(net, l_src_train, l_tgt_train)
            src_test_acc, tgt_test_acc = test(net, l_src_test, l_tgt_test)

            # Evaluate the decision boundary and plot
            # if src_prefix != '  ' or tgt_prefix != '  ' or src_prefix == '**' or tgt_prefix == '++':
            if True:
                # tsne = TSNE(n_components=2, verbose=False, perplexity=50, n_iter=600)
                with torch.no_grad():
                    for j, (data) in enumerate(l_eval_grid):
                        # if ((j + 1) * l_src_test.batch_size) > args.n_test_samples:
                        #     break
                        # data, target = data.to(device), target.to(device)
                        data = data[0].cuda()
                        if net.C.module.use_gumbel:
                            _, output_logits, output = net.discriminator(data)
                        else:
                            _, output_logits = net.discriminator(data)
                            output = F.softmax(output_logits, dim=1)
                        _, labels_pred = output.max(1, keepdim=True)  # get the index of the max log-probability

                        if j == 0:
                            Z = output[:, 0].cpu()
                        else:
                            Z = torch.cat((Z, output[:, 0].cpu()), 0)

            Z_numpy = Z.reshape(64, 64).numpy()
            # plt.contour(XX, YY, Z_numpy, levels=[0.5], cmap='gray')

            # tsne = TSNE(n_components=2, verbose=False, perplexity=50, n_iter=600)
            with torch.no_grad():
                for j, (data, target) in enumerate(l_src_test):
                    if ((j + 1) * l_src_test.batch_size) > args.n_test_samples:
                        break
                    # data, target = data.to(device), target.to(device)
                    data, target = data.cuda(), target.cuda()
                    if net.C.module.use_gumbel:
                        _, output_logits, output = net.discriminator(data)
                    else:
                        _, output_logits = net.discriminator(data)
                        output = F.softmax(output_logits, dim=1)
                    src_test_pred_prob, src_test_pred = output.max(1,
                                                                   keepdim=True)  # get the index of the max log-probability

                    labels_orig = target.cpu().numpy().tolist()
                    labels_pred = src_test_pred.squeeze().cpu().numpy().tolist()
                    pred_prob = src_test_pred_prob.squeeze().cpu().numpy().tolist()
                    if j == 0:
                        # src_mat = output_logits
                        src_mat = data
                        src_pred_mat = output
                        src_labels_orig = labels_orig
                        src_labels_pred = labels_pred
                        src_pred_prob = pred_prob
                        src_label_img = data
                    else:
                        # src_mat = torch.cat((src_mat, output_logits), 0)
                        src_mat = torch.cat((src_mat, data), 0)
                        src_pred_mat = torch.cat((src_pred_mat, output), 0)
                        src_labels_orig.extend(labels_orig)
                        src_labels_pred.extend(labels_pred)
                        src_pred_prob.extend(pred_prob)
                        src_label_img = torch.cat((src_label_img, data), 0)

                # src_labels_orig_str = list(map(str, src_labels_orig))
                # src_labels_pred_str = list(map(str, src_labels_pred))
                # src_metadata = list(map(lambda src_labels_orig_str,
                #                                src_labels_pred_str: 'SRC_TST_Ori' + src_labels_orig_str + '_Pre' + src_labels_pred_str,
                #                         src_labels_orig_str, src_labels_pred_str))
                # # label_img.data.mul_(0.5).add_(0.5)
                # # writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=i)
                #
                # net.writer.add_pr_curve('SRC_PR', torch.tensor(src_labels_orig), torch.tensor(src_labels_pred), global_step=epoch)

                for j, (data, target) in enumerate(l_tgt_test):
                    if ((j + 1) * l_src_test.batch_size) > args.n_test_samples:
                        break
                    # data, target = data.to(device), target.to(device)
                    data, target = data.cuda(), target.cuda()
                    if net.C.module.use_gumbel:
                        _, output_logits, output = net.discriminator(data)
                    else:
                        _, output_logits = net.discriminator(data)
                        output = F.softmax(output_logits, dim=1)
                    tgt_test_pred_pred, tgt_test_pred = output.max(1,
                                                                   keepdim=True)  # get the index of the max log-probability

                    labels_orig = target.cpu().numpy().tolist()
                    labels_pred = tgt_test_pred.squeeze().cpu().numpy().tolist()
                    pred_prob = tgt_test_pred_pred.squeeze().cpu().numpy().tolist()
                    if j == 0:
                        # tgt_mat = output_logits
                        tgt_mat = data
                        tgt_pred_mat = output
                        tgt_labels_orig = labels_orig
                        tgt_labels_pred = labels_pred
                        tgt_pred_prob = pred_prob
                        tgt_label_img = data
                    else:
                        # tgt_mat = torch.cat((tgt_mat, output_logits), 0)
                        tgt_mat = torch.cat((tgt_mat, data), 0)
                        tgt_pred_mat = torch.cat((tgt_pred_mat, output), 0)
                        tgt_labels_orig.extend(labels_orig)
                        tgt_labels_pred.extend(labels_pred)
                        tgt_pred_prob.extend(pred_prob)
                        tgt_label_img = torch.cat((tgt_label_img, data), 0)

                # for j, (data, target) in enumerate(l_tgt_train):
                #     if ((j+1)*l_tgt_train.batch_size) > args.n_test_samples:
                #         break
                #     # data, target = data.to(device), target.to(device)
                #     data, target = data.cuda(), target.cuda()
                #     if net.C.module.use_gumbel:
                #         _, output_logits, output = net.discriminator(data)
                #     else:
                #         _, output_logits = net.discriminator(data)
                #         output = F.softmax(output_logits, dim=1)
                #     tgt_train_pred_pred, tgt_train_pred = output.max(1, keepdim=True)  # get the index of the max log-probability
                #
                #     labels_orig = target.cpu().numpy().tolist()
                #     labels_pred = tgt_train_pred.squeeze().cpu().numpy().tolist()
                #     pred_prob = tgt_train_pred_pred.squeeze().cpu().numpy().tolist()
                #     if j == 0:
                #         # tgt_mat = output_logits
                #         tgt_train_mat = data
                #         tgt_train_pred_mat = output
                #         tgt_train_labels_orig = labels_orig
                #         tgt_train_labels_pred = labels_pred
                #         tgt_train_pred_prob = pred_prob
                #         tgt_train_label_img = data
                #     else:
                #         # tgt_mat = torch.cat((tgt_mat, output_logits), 0)
                #         tgt_train_mat = torch.cat((tgt_train_mat, data), 0)
                #         tgt_train_pred_mat = torch.cat((tgt_train_pred_mat, output), 0)
                #         tgt_train_labels_orig.extend(labels_orig)
                #         tgt_train_labels_pred.extend(labels_pred)
                #         tgt_train_pred_prob.extend(pred_prob)
                #         tgt_train_label_img = torch.cat((tgt_train_label_img, data), 0)

                # tgt_labels_orig_str = list(map(str, tgt_labels_orig))
                # tgt_labels_pred_str = list(map(str, tgt_labels_pred))
                # tgt_metadata = list(map(lambda tgt_labels_orig_str,
                #                                tgt_labels_pred_str: 'TGT_TST_Ori' + tgt_labels_orig_str + '_Pre' + tgt_labels_pred_str,
                #                         tgt_labels_orig_str, tgt_labels_pred_str))
                #
                # net.writer.add_pr_curve('TGT_PR', torch.tensor(tgt_labels_orig), torch.tensor(tgt_labels_pred), global_step=epoch)

                for j, (data, target) in enumerate(l_src_train):
                    if ((j + 1) * l_src_train.batch_size) > args.n_test_samples:
                        break
                    # data, target = data.to(device), target.to(device)
                    data, target = data.cuda(), target.cuda()
                    if net.C.module.use_gumbel:
                        _, output_logits, output = net.discriminator(data)
                    else:
                        _, output_logits = net.discriminator(data)
                        output = F.softmax(output_logits, dim=1)
                    src_train_pred_pred, src_train_pred = output.max(1,
                                                                     keepdim=True)  # get the index of the max log-probability

                    labels_orig = target.cpu().numpy().tolist()
                    labels_pred = src_train_pred.squeeze().cpu().numpy().tolist()
                    pred_prob = src_train_pred_pred.squeeze().cpu().numpy().tolist()
                    if j == 0:
                        # src_mat = output_logits
                        src_train_mat = data
                        src_train_pred_mat = output
                        src_train_labels_orig = labels_orig
                        src_train_labels_pred = labels_pred
                        src_train_pred_prob = pred_prob
                        src_train_label_img = data
                    else:
                        # src_mat = torch.cat((src_mat, output_logits), 0)
                        src_train_mat = torch.cat((src_train_mat, data), 0)
                        src_train_pred_mat = torch.cat((src_train_pred_mat, output), 0)
                        src_train_labels_orig.extend(labels_orig)
                        src_train_labels_pred.extend(labels_pred)
                        src_train_pred_prob.extend(pred_prob)
                        src_train_label_img = torch.cat((src_train_label_img, data), 0)

            # mat = torch.cat((src_mat, tgt_mat), dim=0)
            # pred_mat = torch.cat((src_pred_mat, tgt_pred_mat), dim=0)
            # # metadata = src_metadata + tgt_metadata
            # pred_prob = src_pred_prob + tgt_pred_prob
            #
            # # label_img = torch.cat((src_label_img, tgt_label_img), dim=0)
            # # label_img.data.mul_(0.5).add_(0.5)
            #
            # # net.writer.add_embedding(mat, metadata=metadata, label_img=label_img, n_iter=(epoch * len(tgt_labels_pred)) + j)
            # # net.writer.add_embedding(mat, metadata=metadata, global_step=epoch)
            #
            # # tsne_results = tsne.fit_transform(mat.cpu().numpy())
            # tsne_results = mat.cpu().numpy()
            # pred_mat = pred_mat.cpu().numpy()

            # with torch.no_grad():
            #     for j, (data, target) in enumerate(l_src_train):
            #         if ((j + 1) * l_src_train.batch_size) > args.n_test_samples:
            #             break
            #         # data, target = data.to(device), target.to(device)
            #         data, target = data.cuda(), target.cuda()
            #         if net.C.module.use_gumbel:
            #             _, output_logits, output = net.discriminator(data)
            #         else:
            #             _, output_logits = net.discriminator(data)
            #             output = F.softmax(output_logits, dim=1)
            #         src_train_pred_pred, src_train_pred = output.max(1,
            #                                                          keepdim=True)  # get the index of the max log-probability
            #
            #         labels_orig = target.cpu().numpy().tolist()
            #         labels_pred = src_train_pred.squeeze().cpu().numpy().tolist()
            #         pred_prob = src_train_pred_pred.squeeze().cpu().numpy().tolist()
            #         if j == 0:
            #             # src_mat = output_logits
            #             src_train_mat = data
            #             src_train_pred_mat = output
            #             src_train_labels_orig = labels_orig
            #             src_train_labels_pred = labels_pred
            #             src_train_pred_prob = pred_prob
            #             src_train_label_img = data
            #         else:
            #             # src_mat = torch.cat((src_mat, output_logits), 0)
            #             src_train_mat = torch.cat((src_train_mat, data), 0)
            #             src_train_pred_mat = torch.cat((src_train_pred_mat, output), 0)
            #             src_train_labels_orig.extend(labels_orig)
            #             src_train_labels_pred.extend(labels_pred)
            #             src_train_pred_prob.extend(pred_prob)
            #             src_train_label_img = torch.cat((src_train_label_img, data), 0)

            # src_labels_orig_str = list(map(str, src_labels_orig))
            # src_labels_pred_str = list(map(str, src_labels_pred))
            # src_metadata = list(map(lambda src_labels_orig_str,
            #                                src_labels_pred_str: 'TGT_TST_Ori' + src_labels_orig_str + '_Pre' + src_labels_pred_str,
            #                         src_labels_orig_str, src_labels_pred_str))
            #
            # net.writer.add_pr_curve('TGT_PR', torch.tensor(src_labels_orig), torch.tensor(src_labels_pred), global_step=epoch)

            mat = torch.cat((src_mat, tgt_mat, src_train_mat), dim=0)
            pred_mat = torch.cat((src_pred_mat, tgt_pred_mat, src_train_pred_mat), dim=0)
            # metadata = src_metadata + src_metadata
            pred_prob = src_pred_prob + tgt_pred_prob + src_train_pred_prob

            tsne_results = mat.cpu().numpy()
            pred_mat = pred_mat.cpu().numpy()

            if src_test_acc >= best_src_test_acc:
                # log_str += log('*** Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
                #     epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
                src_prefix = '**'
                best_src_test_acc = src_test_acc
                net.save_model(suffix='_src_test')
            else:
                # log_str += log('    Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
                #         epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
                src_prefix = '  '

            if tgt_test_acc >= best_tgt_test_acc:
                # log_str += log('+++ Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
                #     epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
                tgt_prefix = '++'
                best_tgt_test_acc = tgt_test_acc
                net.save_model(suffix='_tgt_test')

                # # tsne = TSNE(n_components=2, verbose=False, perplexity=50, n_iter=600)
                # with torch.no_grad():
                #     for j, (data, target) in enumerate(l_src_test):
                #         if ((j+1)*l_src_test.batch_size) > args.n_test_samples:
                #             break
                #         # data, target = data.to(device), target.to(device)
                #         data, target = data.cuda(), target.cuda()
                #         if net.C.module.use_gumbel:
                #             _, output_logits, output = net.discriminator(data)
                #         else:
                #             _, output_logits = net.discriminator(data)
                #             output = F.softmax(output_logits, dim=1)
                #         src_test_pred_prob, src_test_pred = output.max(1, keepdim=True)  # get the index of the max log-probability
                #
                #         labels_orig = target.cpu().numpy().tolist()
                #         labels_pred = src_test_pred.squeeze().cpu().numpy().tolist()
                #         pred_prob = src_test_pred_prob.squeeze().cpu().numpy().tolist()
                #         if j == 0:
                #             # src_mat = output_logits
                #             src_mat = data
                #             src_pred_mat = output
                #             src_labels_orig = labels_orig
                #             src_labels_pred = labels_pred
                #             src_pred_prob = pred_prob
                #             src_label_img = data
                #         else:
                #             # src_mat = torch.cat((src_mat, output_logits), 0)
                #             src_mat = torch.cat((src_mat, data), 0)
                #             src_pred_mat = torch.cat((src_pred_mat, output), 0)
                #             src_labels_orig.extend(labels_orig)
                #             src_labels_pred.extend(labels_pred)
                #             src_pred_prob.extend(pred_prob)
                #             src_label_img = torch.cat((src_label_img, data), 0)
                #
                #     # src_labels_orig_str = list(map(str, src_labels_orig))
                #     # src_labels_pred_str = list(map(str, src_labels_pred))
                #     # src_metadata = list(map(lambda src_labels_orig_str,
                #     #                                src_labels_pred_str: 'SRC_TST_Ori' + src_labels_orig_str + '_Pre' + src_labels_pred_str,
                #     #                         src_labels_orig_str, src_labels_pred_str))
                #     # # label_img.data.mul_(0.5).add_(0.5)
                #     # # writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=i)
                #     #
                #     # net.writer.add_pr_curve('SRC_PR', torch.tensor(src_labels_orig), torch.tensor(src_labels_pred), global_step=epoch)
                #
                #     for j, (data, target) in enumerate(l_tgt_test):
                #         if ((j+1)*l_src_test.batch_size) > args.n_test_samples:
                #             break
                #         # data, target = data.to(device), target.to(device)
                #         data, target = data.cuda(), target.cuda()
                #         if net.C.module.use_gumbel:
                #             _, output_logits, output = net.discriminator(data)
                #         else:
                #             _, output_logits = net.discriminator(data)
                #             output = F.softmax(output_logits, dim=1)
                #         tgt_test_pred_pred, tgt_test_pred = output.max(1, keepdim=True)  # get the index of the max log-probability
                #
                #         labels_orig = target.cpu().numpy().tolist()
                #         labels_pred = tgt_test_pred.squeeze().cpu().numpy().tolist()
                #         pred_prob = tgt_test_pred_pred.squeeze().cpu().numpy().tolist()
                #         if j == 0:
                #             # tgt_mat = output_logits
                #             tgt_mat = data
                #             tgt_pred_mat = output
                #             tgt_labels_orig = labels_orig
                #             tgt_labels_pred = labels_pred
                #             tgt_pred_prob = pred_prob
                #             tgt_label_img = data
                #         else:
                #             # tgt_mat = torch.cat((tgt_mat, output_logits), 0)
                #             tgt_mat = torch.cat((tgt_mat, data), 0)
                #             tgt_pred_mat = torch.cat((tgt_pred_mat, output), 0)
                #             tgt_labels_orig.extend(labels_orig)
                #             tgt_labels_pred.extend(labels_pred)
                #             tgt_pred_prob.extend(pred_prob)
                #             tgt_label_img = torch.cat((tgt_label_img, data), 0)
                #
                #     # for j, (data, target) in enumerate(l_tgt_train):
                #     #     if ((j+1)*l_tgt_train.batch_size) > args.n_test_samples:
                #     #         break
                #     #     # data, target = data.to(device), target.to(device)
                #     #     data, target = data.cuda(), target.cuda()
                #     #     if net.C.module.use_gumbel:
                #     #         _, output_logits, output = net.discriminator(data)
                #     #     else:
                #     #         _, output_logits = net.discriminator(data)
                #     #         output = F.softmax(output_logits, dim=1)
                #     #     tgt_train_pred_pred, tgt_train_pred = output.max(1, keepdim=True)  # get the index of the max log-probability
                #     #
                #     #     labels_orig = target.cpu().numpy().tolist()
                #     #     labels_pred = tgt_train_pred.squeeze().cpu().numpy().tolist()
                #     #     pred_prob = tgt_train_pred_pred.squeeze().cpu().numpy().tolist()
                #     #     if j == 0:
                #     #         # tgt_mat = output_logits
                #     #         tgt_train_mat = data
                #     #         tgt_train_pred_mat = output
                #     #         tgt_train_labels_orig = labels_orig
                #     #         tgt_train_labels_pred = labels_pred
                #     #         tgt_train_pred_prob = pred_prob
                #     #         tgt_train_label_img = data
                #     #     else:
                #     #         # tgt_mat = torch.cat((tgt_mat, output_logits), 0)
                #     #         tgt_train_mat = torch.cat((tgt_train_mat, data), 0)
                #     #         tgt_train_pred_mat = torch.cat((tgt_train_pred_mat, output), 0)
                #     #         tgt_train_labels_orig.extend(labels_orig)
                #     #         tgt_train_labels_pred.extend(labels_pred)
                #     #         tgt_train_pred_prob.extend(pred_prob)
                #     #         tgt_train_label_img = torch.cat((tgt_train_label_img, data), 0)
                #
                #     # tgt_labels_orig_str = list(map(str, tgt_labels_orig))
                #     # tgt_labels_pred_str = list(map(str, tgt_labels_pred))
                #     # tgt_metadata = list(map(lambda tgt_labels_orig_str,
                #     #                                tgt_labels_pred_str: 'TGT_TST_Ori' + tgt_labels_orig_str + '_Pre' + tgt_labels_pred_str,
                #     #                         tgt_labels_orig_str, tgt_labels_pred_str))
                #     #
                #     # net.writer.add_pr_curve('TGT_PR', torch.tensor(tgt_labels_orig), torch.tensor(tgt_labels_pred), global_step=epoch)
                #
                # mat = torch.cat((src_mat, tgt_mat), dim=0)
                # pred_mat = torch.cat((src_pred_mat, tgt_pred_mat), dim=0)
                # # metadata = src_metadata + tgt_metadata
                # pred_prob = src_pred_prob + tgt_pred_prob
                #
                # # label_img = torch.cat((src_label_img, tgt_label_img), dim=0)
                # # label_img.data.mul_(0.5).add_(0.5)
                #
                # # net.writer.add_embedding(mat, metadata=metadata, label_img=label_img, n_iter=(epoch * len(tgt_labels_pred)) + j)
                # # net.writer.add_embedding(mat, metadata=metadata, global_step=epoch)
                #
                # # tsne_results = tsne.fit_transform(mat.cpu().numpy())
                # tsne_results = mat.cpu().numpy()
                # pred_mat = pred_mat.cpu().numpy()

                # # # Plot TSNE

                fig = plt.figure()
                # plt.subplot(122)
                # plt.title("Domain adapted plot", fontsize='small')

                x, y = tsne_results[:, 0], tsne_results[:, 1]
                # plt.contour(x, y, np.expand_dims(pred_mat[:, 0], axis=0), levels=0.5, cmap='gray')
                cax = plt.scatter(tsne_results[:len(src_labels_orig), 0], tsne_results[:len(src_labels_orig), 1],
                                  c=torch.tensor(src_labels_pred).numpy(), alpha=1.0,
                                  cmap=plt.cm.get_cmap("jet", n_classes), marker='x',
                                  label='SRC - {} - #{}'.format(d_src_test.dataset_name.upper(), len(src_labels_orig)))
                cax = plt.scatter(tsne_results[len(src_labels_orig):len(src_labels_orig)+len(tgt_labels_orig), 0], tsne_results[len(src_labels_orig):len(src_labels_orig)+len(tgt_labels_orig), 1],
                                  c=torch.tensor(tgt_labels_pred).numpy(), alpha=1.0,
                                  cmap=plt.cm.get_cmap("jet", d_tgt_test.n_classes), marker='+',
                                  label='TGT - {} - #{}'.format(d_tgt_test.dataset_name.upper(), len(tgt_labels_orig)))
                plt.legend(loc='upper left')
                plt.contour(XX, YY, Z_numpy, levels=[0.5], cmap='gray')
                # plt.legend(loc='upper right')
                fig.colorbar(cax, extend='min', ticks=np.arange(0, n_classes))
                # plt.text(0, 0.1, r'$\delta$',
                #          {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
                #           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
                # plt.figtext(.5, .9, '{} : ACCURACIES\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(args.exp.upper(), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=10)
                # plt.tight_layout()
                plt.axis('off')
                os.system('mkdir -p {}/plots'.format(net.args.ckpt_dir))
                plt.savefig('{}/plots/tsne/src_{}_seed_{:04d}_ss_tsne_epoch{:03d}.png'.format(net.args.ckpt_dir, src_name, seed, epoch))
                plt.title('{} : ACCURACIES : EPOCH {}\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(args.exp.upper(), epoch, src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=8)
                plt.savefig('{}/plots/tsne_acc/src_{}_seed_{:04d}_ss_tsne_acc_epoch{:03d}.png'.format(net.args.ckpt_dir, src_name, seed, epoch))
                # net.writer.add_figure('{}_tsne'.format(args.exp), fig, global_step=(epoch * len(tgt_labels_pred)) + j)
                plt.tight_layout()
                net.writer.add_figure('src_{}_seed_{:04d}_ss_tsne'.format(src_name, seed), fig, global_step=epoch)

            else:
                # log_str += log('    Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
                #         epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
                tgt_prefix = '  '

            # Plot contour with heatmaps
            fig = plt.figure()
            plt.contourf(XX, YY, Z_numpy, 20, cmap='coolwarm')  # PuOr, binary, RdGy  # contour plots wth prob
            plt.colorbar()  # contour plots wth prob
            # plt.contour(XX, YY, Z_numpy, levels=[0.5], cmap='gray')
            x, y = tsne_results[:, 0], tsne_results[:, 1]
            # plt.contour(x, y, np.expand_dims(pred_mat[:, 0], axis=0), levels=0.5, cmap='gray')
            cax = plt.scatter(tsne_results[:len(src_labels_orig), 0], tsne_results[:len(src_labels_orig), 1],
                              c=torch.tensor(src_labels_orig).numpy(), alpha=1.0,
                              cmap=plt.cm.get_cmap("jet", n_classes),
                              marker='x',
                              label='SRC - {} - #{}'.format(d_src_test.dataset_name.upper(), len(src_labels_orig)))
            cax = plt.scatter(tsne_results[len(src_labels_orig):len(src_labels_orig)+len(tgt_labels_orig), 0], tsne_results[len(src_labels_orig):len(src_labels_orig)+len(tgt_labels_orig), 1],
                              c=torch.tensor(tgt_labels_orig).numpy(), alpha=1.0,
                              cmap=plt.cm.get_cmap("jet", d_tgt_test.n_classes),
                              marker='+',
                              label='TGT - {} - #{}'.format(d_tgt_test.dataset_name.upper(), len(tgt_labels_orig)))
            cax = plt.scatter(tsne_results[len(src_labels_orig)+len(tgt_labels_orig):, 0], tsne_results[len(src_labels_orig)+len(tgt_labels_orig):, 1],
                              c=torch.tensor(src_train_labels_orig).numpy(), alpha=1.0,
                              cmap=plt.cm.get_cmap("jet", d_src_train.n_classes),
                              marker='*',
                              label='TGT - {} aligned - #{}'.format(d_tgt_train.dataset_name.upper(), len(src_train_labels_orig)))
            plt.legend(loc='upper left')
            # plt.legend(loc='upper right')
            # fig.colorbar(cax, extend='min', ticks=np.arange(0, n_classes))
            # plt.text(0, 0.1, r'$\delta$',
            #          {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
            #           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
            # plt.figtext(.5, .9, '{} : ACCURACIES\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(args.exp.upper(), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=10)
            # plt.tight_layout()
            plt.contour(XX, YY, Z_numpy, levels=[0.5], cmap='gray')
            plt.axis('off')
            os.system('mkdir -p {}/plots'.format(net.args.ckpt_dir))
            plt.savefig('{}/plots/contour/src_{}_seed_{:04d}_ss_contour_epoch{:03d}.png'.format(net.args.ckpt_dir, src_name, seed, epoch))
            plt.title('{} : ACCURACIES : EPOCH {}\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(
                args.exp.upper(), epoch, src_train_acc / (l_src_train.__len__() * batch_size),
                tgt_train_acc / (l_tgt_train.__len__() * batch_size), src_test_acc / len(d_src_test),
                tgt_test_acc / len(d_tgt_test)), fontsize=8)
            plt.savefig('{}/plots/contour_acc/src_{}_seed_{:04d}_ss_contour_acc_epoch{:03d}.png'.format(net.args.ckpt_dir, src_name, seed, epoch))
            # net.writer.add_figure('{}_tsne'.format(args.exp), fig, global_step=(epoch * len(tgt_labels_pred)) + j)
            plt.tight_layout()
            net.writer.add_figure('src_{}_seed_{:04d}_ss_contour'.format(src_name, seed), fig, global_step=epoch)

            # log_str += log('{}{} Epoch:{:03d}/{:03d} #batch:{:03d} took {:03.2f}m - Loss:{:03.5f}, SRC_TRAIN_ACC = {:2.2%}, TGT_TRAIN_ACC = {:2.2%}, SRC_TEST_ACC = {:2.2%}, TGT_TEST_ACC = {:2.2%}'.format(
            #     src_prefix, tgt_prefix, epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
            log('{}{} E:{:03d}/{:03d} #B:{:03d}, t={:06.2f}m, L={:07.4f}, ACC : S_TRN= {:5.2%}, T_TRN= {:5.2%}, S_TST= {:5.2%}, T_TST= {:5.2%}'.format(
                src_prefix, tgt_prefix, epoch, num_epochs, i+1, run_time, epoch_loss/(i+1), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)))

            net.writer.add_scalar('Accuracy/train/src', src_train_acc/(l_src_train.__len__() * batch_size), epoch)
            net.writer.add_scalar('Accuracy/train/tgt', tgt_train_acc/(l_tgt_train.__len__() * batch_size), epoch)
            net.writer.add_scalar('Accuracy/test/src', src_test_acc/len(d_src_test), epoch)
            net.writer.add_scalar('Accuracy/test/tgt', tgt_test_acc/len(d_tgt_test), epoch)

            net.writer.add_text('epoch log', '{}{} E:{:03d}/{:03d} #B:{:03d}, t={:06.2f}m, L={:07.4f}, ACC : S_TRN= {:5.2%}, T_TRN= {:5.2%}, S_TST= {:5.2%}, T_TST= {:5.2%}'.format(
                src_prefix, tgt_prefix, epoch, num_epochs, i+1, run_time, epoch_loss/(i+1), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), epoch)
            os.system('cp {0} {1}/Log/'.format(log_file, net.args.ckpt_dir))
            # if use_scheduler:
            #     scheduler_E.step(src_test_acc)
            #     scheduler_C.step(src_test_acc)
            #     scheduler_GS.step(src_test_acc)
            #     scheduler_GT.step(src_test_acc)
            # else:
            #     lr = args.learning_rate * (args.lr_decay_rate ** (epoch // args.lr_decay_period))
            #     for param_group in net.optimizerC.param_groups:
            #         param_group['lr'] = lr
            #     for param_group in net.optimizerE.param_groups:
            #         param_group['lr'] = lr
            #     for param_group in net.optimizerGS.param_groups:
            #         param_group['lr'] = lr
            #     for param_group in net.optimizerGT.param_groups:
            #         param_group['lr'] = lr

        net.save_model()
        net.writer.close()

    except Exception as e:     # most generic exception you can catch
        log('Something went horribly wrong !!!')
        n_iter = 0
        log('Error : {}'.format(str(e)))
        net.writer.add_text('epoch log', 'Something went horribly wrong !!!', n_iter + 1)
        net.writer.add_text('epoch log', 'Error : {}'.format(str(e)), n_iter + 2)
        os.system('mv {0} {0}.err'.format(log_file))
        os.system('mv {1}/Log/{0} {1}/Log/{0}.err'.format(log_file, net.args.ckpt_dir))
        os.system('mv {0} {0}.err'.format(net.args.ckpt_dir))
        net.writer.close()
        raise


if __name__ == '__main__':
    # Get argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['toy_blobs'], default='toy_blobs', help='experiment to run')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate (Adam)')
    parser.add_argument('--num_epochs', type=int, default=600, help='number of epochs : (default=3000)')
    parser.add_argument('--ramp', type=int, default=0, help='ramp for epochs')
    parser.add_argument('--buffer_size', type=int, default=1000, help='length of the buffer for latent feature selection : (default=10000)')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size : (default=64)')
    parser.add_argument('--seed', type=int, default=22, help='random seed (0 for time-based)')  # seed = 310, 1211, 1511, 2082, 2234, 211, 2737, 22, 3001
    # 22, 3001, 3234,
    parser.add_argument('--log_file', type=str, default='', help='log file path (none to disable)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
    parser.add_argument('--gpus', type=str, default='0,1', help='using gpu device id')
    # parser.add_argument('--gpus', type=str, default='3,2,1,0', help='using gpu device id')
    parser.add_argument('--plot_interval', type=int, default=60, help='Number of plots required to save every iteration')
    parser.add_argument('--img_dir', type=str, default='./images', help='path to save images')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='path to save logs')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint', help='path to save model checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default='', help='path to checkpoint')

    # Hyperparameter search for reproducing - hp448.sh
    parser.add_argument('--use_sampler', action='store_true', default=True, help='use sampler for dataloader (default : False)')
    parser.add_argument('--use_gen_sqrt', action='store_true', default=False, help='use squareroot for MMD loss (default : False)')
    parser.add_argument('--train_GnE', action='store_true', default=False, help='train encoder with generator training (default : False)')
    parser.add_argument('--network_type', type=str, default='se', help='type of network [se|mcd|dade]')
    parser.add_argument('--use_tied_gen', action='store_true', default=False, help='use a single generator for source and target domain (default : False)')
    parser.add_argument('--epoch_size', default='large', choices=['large', 'small', 'source', 'target'], help='epoch size is either that of the smallest dataset, the largest, the source, or the target (default=source)')
    parser.add_argument('--use_drop', action='store_true', default=True, help='use dropout for classifier (default : False)')
    parser.add_argument('--use_bn', action='store_true', default=False, help='use batchnorm for classifier (default : False)')
    parser.add_argument('--lr_decay_type', choices=['scheduler', 'geometric', 'none'], default='geometric', help='lr_decay_type (default=none)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.6318, help='learning rate (Adam)')
    parser.add_argument('--lr_decay_period', type=float, default=50, help='learning rate (Adam)')
    parser.add_argument('--weight_init', type=str, default='none', help='type of weight initialization')
    parser.add_argument('--use_gumbel', action='store_true', default=False, help='use Gumbel softmax for label selection (default : False)')
    parser.add_argument('--n_test_samples', type=int, default=1000, help='number of test samples used for plotting t-SNE')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension of toy dataset')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--n_samples', type=int, default=4000, help='number of samples per domain')
    parser.add_argument('--split_size', type=float, default=0.5, help='train/test split ratio')

    # class0_centroid_radius = 7.5
    # class1_centroid_radius = 2.5
    # domain0_orient_angle = 90  # in degrees
    # domain1_orient_angle = 45  # in degrees
    parser.add_argument('--is_centroid_based', action='store_true', default=False, help='use centroid based dataset creator')
    parser.add_argument('--class0_centroid_radius', '-c0r', type=float, default=10, help='radius of centroids for class 0')
    parser.add_argument('--class1_centroid_radius', '-c1r', type=float, default=5, help='radius of centroids for class 1')
    parser.add_argument('--domain0_orient_angle', '-d0a', type=float, default=90, help='orientation of domain 0')
    parser.add_argument('--domain1_orient_angle', '-d1a', type=float, default=45, help='orientation of domain 1')
    parser.add_argument('--swap_domain', action='store_true', default=False, help='Swap source and target domains')

    #     parser = get_dade_args(parser)
    args = parser.parse_args()
    print(args)

    experiment(args)
    # experiment()
    '''
    Run toy dataset experiment
    
    convert -delay 10 -loop 0 `ls -v toy_blobs_tsne_epoch*_acc.png` toy_dataset_cuda.gif
    convert -delay 10 -loop 0 `ls -v toy_blobs_contour_epoch*_acc.png' ../../../gifs/toy_dataset_cuda_3234.gif
    convert -delay 10 -loop 0 toy_blobs_contour_epoch*_acc.png toy_dataset_cuda_3234.gif
    crontab -e */10 * * * * rsync -a  sourabh@10.192.30.11:/home/sourabh/prj/devilda_toy/checkpoint/ /home/sourabh/prj/devilda_toy/checkpoint/
    '''
