import torch
import torch.cuda
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dutils
import torchvision.utils as vutils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.manifold import TSNE
import matplotlib
import lr_schedule

matplotlib.use('Agg')
# matplotlib.use('WXAgg')
# matplotlib.use('GTKAgg')
# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

# SE Imports
import numpy as np
import cmdline_helpers
from mmd import mix_rbf_mmd2
import socket
import datetime
import math
import random
import os
import sys
import timeit
from itertools import cycle

# MMD GAN Imports

# from model.build_network import Network
from util import get_data
from model.build_network import Network

# os.environ["CUDA_VISIBLE_DEVICES"] = '3,2,1,0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,3,2,1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,0,3,2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,1,0,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '7,6,5,4'
# os.environ["CUDA_VISIBLE_DEVICES"] = '6,5,4,7'
# os.environ["CUDA_VISIBLE_DEVICES"] = '5,4,7,6
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,7,6,5'

# os.environ["CUDA_VISIBLE_DEVICES"] = '3,2,1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,1,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,3,2'

# os.environ["CUDA_VISIBLE_DEVICES"] = '2,1,0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,2,1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,0,2'

# os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def experiment(args):
    exp = args.exp
    # learning_rate = args.learning_rate
    # num_epochs = args.num_epochs
    batch_size = args.batch_size
    epoch_size = args.epoch_size = 'dataplots'
    seed = args.seed
    log_file = args.log_file
    workers = args.workers
    image_size = args.image_size
    nc = args.nc
    # nz = args.nz
    use_gpu = True
    gpus = args.gpus
    img_dir = args.img_dir
    logs_dir = args.logs_dir
    # ckpt_dir = args.ckpt_dir
    # plot_interval = args.plot_interval
    experiment = args.exp

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Some variable hardcoding - Only for development
    # seed = 1126
    seed = args.seed
    simul_train_src_tgt = True
    # use_scheduler = args.use_scheduler  # True/False
    use_sampler = True  # True/False

    machinename = socket.gethostname()
    # hostname = timestamp = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    # hostname = timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
    hostname = timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    absolute_pyfile_path = os.path.abspath(sys.argv[0])
    args.absolute_pyfile_path = os.path.abspath(sys.argv[0])

    absolute_base_path = os.path.dirname(absolute_pyfile_path)
    args.absolute_base_path = os.path.dirname(absolute_pyfile_path)

    args.dataroot = os.path.expanduser(args.dataroot)
    dataroot = os.path.join(absolute_base_path, args.dataroot)
    dataset_path = os.path.join(absolute_base_path, args.dataroot, args.dataset)
    args.dataroot = os.path.join(absolute_base_path, args.dataroot)
    args.dataset_path = os.path.join(absolute_base_path, args.dataroot, args.dataset)

    args.logs_dir = os.path.expanduser(args.logs_dir)
    # logs_dir = args.logs_dir
    logs_dir = os.path.join(absolute_base_path, args.logs_dir)
    args.logs_dir = logs_dir

    log_num = 0
    log_file = logs_dir + '/' + hostname + '_' + machinename + '_' + args.exp + '_' + str(args.seed) + '.txt'

    # Setup logfile to store output logs
    if log_file is not None:
        while os.path.exists(log_file):
            log_num += 1
            log_file = '{0}/{3}_{1}_{2}_{4}.txt'.format(args.logs_dir, 'datasplots', log_num, hostname, args.epoch_size)
        # return
    args.log_file = log_file

    args.img_dir = os.path.expanduser(args.img_dir)
    img_dir = os.path.join(absolute_base_path, args.img_dir, args.exp) + '_' + str(log_num)
    args.img_dir = img_dir

    # Create folder to save log files
    os.system('mkdir -p {0}'.format(logs_dir))  # create folder to store log files

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

    n_iter = 0
    log_str = ''
    log_str += log('\n')
    log_str += log('Output log file {0} created'.format(log_file))
    log_str += log('File used to run the experiment : {0}'.format(absolute_pyfile_path))
    log_str += log('Output image files are stored in {0} directory'.format(img_dir))

    # Report setttings
    log_str += log('Settings: {}'.format(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))])))

    num_gpu = len(args.gpus.split(','))
    log_str += log('num_gpu: {0}, GPU-ID: {1}'.format(num_gpu, args.gpus))

    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
        # if num_gpu < 4:
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    dataset_list = ['usps', 'mnist',
                     'svhn', 'syndigits',
                     'cifar9', 'stl9',
                     'gtsrb', 'synsigns',
                     'amazon', 'dslr', 'webcam',
                     # 'cifar10', 'cifar100',
                     # 'fmnist', 'emnist',
                     # 'lsun', 'imagenet',
                     # 'folder', 'lfw',
                     ]

    transform = None

    # Get datasets
    if exp == 'usps_mnist':
        args.nc = 1
        args.image_size = 28
        args.dataset = 'usps'
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = 'mnist'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp == 'mnist_usps':
        args.nc = 1
        args.image_size = 28
        args.dataset = 'mnist'
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = 'usps'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp == 'svhn_mnist':
        # args.nc = 1
        # args.image_size = 28
        if args.network_type == 'dade':
            args.nc = 1
            args.image_size = 28
        else:
            args.nc = 3
            args.image_size = 32
        args.dataset = 'svhn'
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = 'mnist'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp == 'mnist_svhn':
        # args.nc = 1
        # args.image_size = 32
        # args.image_size = 28
        if args.network_type == 'dade':
            args.nc = 1
            args.image_size = 28
        else:
            args.nc = 3
            args.image_size = 32
        args.dataset = 'mnist'
        d_src_train_notinverted, _ = get_data(args, transform=transform, train_flag=False)
        d_src_train_inverted, d_src_test = get_data(args, transform=transform, train_flag=True)
        d_src_train = torch.utils.data.ConcatDataset([d_src_train_inverted, d_src_train_notinverted])
        d_src_train.dataset_name = 'mnist'
        d_src_train.n_classes = 10
        d_src_train.transform = d_src_train_notinverted.transform
        args.dataset = 'svhn'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp == 'cifar_stl':
        args.nc = 3
        args.image_size = 32
        args.dataset = 'cifar9'
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = 'stl9'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp == 'stl_cifar':
        args.nc = 3
        args.image_size = 32
        args.dataset = 'stl9'
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = 'cifar9'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp == 'syndigits_svhn':
        # args.nc = 1
        # # args.image_size = 28
        # args.image_size = 32
        args.nc = 3
        args.image_size = 32
        args.dataset = 'syndigits'
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = 'svhn'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp == 'svhn_syndigits':
        # args.nc = 1
        # args.image_size = 28
        args.nc = 3
        args.image_size = 32
        args.dataset = 'svhn'
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = 'syndigits'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp == 'synsigns_gtsrb':
        args.nc = 3
        args.image_size = 40
        # args.nc = 3
        # args.image_size = 32
        args.dataset = 'synsigns'
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = 'gtsrb'
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    elif exp in {'amazon_dslr', 'amazon_webcam', 'dslr_amazon', 'dslr_webcam', 'webcam_amazon', 'webcam_dslr'}:
        args.nc = 3
        args.image_size = 224
        args.dataset = exp.split('_')[0]
        d_src_train, d_src_test = get_data(args, transform=transform, train_flag=True)
        args.dataset = exp.split('_')[1]
        d_tgt_train, d_tgt_test = get_data(args, transform=transform, train_flag=False)
    else:
        log_str += log('Error : Unknown experiment type \'{}\''.format(exp))
        raise ValueError('Unknown experiment type \'{}\''.format(exp))

    log_str += log('\nSRC : {}: train: count={}, X.shape={} test: count={}, X.shape={}'.format(
        d_src_train.dataset_name.upper(), d_src_train.__len__(), d_src_train[0][0].shape,
        d_src_test.__len__(), d_src_test[0][0].shape))
    log_str += log('TGT : {}: train: count={}, X.shape={} test: count={}, X.shape={}'.format(
        d_tgt_train.dataset_name.upper(), d_tgt_train.__len__(), d_tgt_train[0][0].shape,
        d_tgt_test.__len__(), d_tgt_test[0][0].shape))

    n_classes = d_src_train.n_classes

    log_str += log('\nTransformations for SRC and TGT datasets ...')
    log_str += log('SRC : {0} - transformation : {1}'.format(d_src_train.dataset_name.upper(), d_src_train.transform))
    log_str += log('TGT : {0} - transformation : {1}'.format(d_tgt_train.dataset_name.upper(), d_tgt_train.transform))
    log_str += log('\nNumber of classes : {}'.format(n_classes))
    log_str += log('\nLoaded  Source and Target data respectively')

    if exp == 'synsigns_gtsrb':
        image_grid_n_row = 10
    elif exp in {'amazon_dslr', 'amazon_webcam', 'dslr_amazon', 'dslr_webcam', 'webcam_amazon', 'webcam_dslr'}:
        image_grid_n_row = 8
    else:
        image_grid_n_row = 6

    image_grid = torch.zeros((n_classes * 2, args.nc, args.image_size, args.image_size), requires_grad=False)

    try:
        d_src_train_labels = d_src_train.train_labels.numpy()
    except:
        d_src_train = d_src_train_notinverted
        d_src_train_labels = d_src_train.train_labels.numpy()

    labels, src_indices = np.unique(d_src_train_labels, return_index=True)

    d_tgt_train_labels = d_tgt_train.train_labels.numpy()
    labels, tgt_indices = np.unique(d_tgt_train_labels, return_index=True)

    for i, (src_index, tgt_index) in enumerate(zip(np.nditer(src_indices), np.nditer(tgt_indices))):
        image_grid[2*i, :, :, :] = d_src_train[src_index][0]
        image_grid[2*i+1, :, :, :] = d_tgt_train[tgt_index][0]

    image_grid.data.mul_(0.5).add_(0.5)
    img_grid = vutils.make_grid(image_grid, nrow=image_grid_n_row)  # make an square grid of images and plot
    vutils.save_image(img_grid, '{}.png'.format(log_file[:-4]))


if __name__ == '__main__':
    # Get argument
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['usps_mnist', 'mnist_usps',
                                          'svhn_mnist', 'mnist_svhn',
                                          'cifar_stl', 'stl_cifar',
                                          'syndigits_svhn', 'svhn_syndigits',
                                          'synsigns_gtsrb', 'gtsrb_synsigns',
                                          'amazon_dslr', 'amazon_webcam',
                                          'dslr_amazon', 'dslr_webcam',
                                          'webcam_amazon', 'webcam_dslr',
                                          ], default='usps_mnist', help='experiment to run')
    parser.add_argument('--dataset', choices=['usps', 'mnist',
                                              'svhn', 'syndigits',
                                              'cifar9', 'stl9',
                                              'gtsrb', 'synsigns',
                                              'cifar10', 'cifar100',
                                              'fmnist', 'emnist',
                                              'lsun', 'imagenet',
                                              'folder', 'lfw'
                                              ], default='mnist',
                        help=['usps | mnist | svhn | syndigits | cifar9 | stl9 | gtsrb | synsigns | cifar10 | cifar100',
                              'fmnist | emnist | lsun | imagenet | folder | lfw'])
    parser.add_argument('--dataroot', type=str, default='./data', help='path to args.dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='mini-batch size')
    parser.add_argument('--seed', type=int, default=1126, help='random seed (0 for time-based)')
    parser.add_argument('--log_file', type=str, default='', help='log file path (none to disable)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--image_size', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='number of channel')
    parser.add_argument('--gpus', type=str, default='2,1', help='using gpu device id')
    parser.add_argument('--img_dir', type=str, default='./dataplots', help='path to save images')
    parser.add_argument('--logs_dir', type=str, default='./dataplots', help='path to save logs')
    args = parser.parse_args()

    exps = [
        'usps_mnist', 'mnist_usps',
        'svhn_mnist', 'mnist_svhn',
        'cifar_stl', 'stl_cifar',
        'syndigits_svhn', 'svhn_syndigits',
        'synsigns_gtsrb', # 'gtsrb_synsigns',
        'amazon_dslr', 'amazon_webcam',
        'dslr_amazon', 'dslr_webcam',
        'webcam_amazon', 'webcam_dslr',
    ]

    for exp in exps:
        args.exp = exp
        time.sleep(1)
        experiment(args)
