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


# def test(net, d_src_test, d_tgt_test):
#     # discriminator.eval()
#     net.E.eval()
#     net.C.eval()
#     l_src_test = dutils.DataLoader(d_src_test,
#                                    batch_size=128,
#                                    shuffle=True,
#                                    num_workers=int(4),
#                                    drop_last=False)
#
#     l_tgt_test = dutils.DataLoader(d_tgt_test,
#                                    batch_size=128,
#                                    shuffle=True,
#                                    num_workers=int(4),
#                                    drop_last=False)
#     src_test_correct = 0
#     tgt_test_correct = 0
#     with torch.no_grad():
#         for data, target in l_src_test:
#             # data, target = data.to(device), target.to(device)
#             data, target = data.cuda(), target.cuda()
#             _, output_logits = net.discriminator(data)
#             output = F.softmax(output_logits, dim=1)
#             src_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#             src_test_correct += src_test_pred.eq(target.view_as(src_test_pred)).sum().item()
#
#         for data, target in l_tgt_test:
#             # data, target = data.to(device), target.to(device)
#             data, target = data.cuda(), target.cuda()
#             _, output_logits = net.discriminator(data)
#             output = F.softmax(output_logits, dim=1)
#             tgt_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#             tgt_test_correct += tgt_test_pred.eq(target.view_as(tgt_test_pred)).sum().item()
#
#     # print('\nSRC Test Acc: {:.0f}%, TGT Test Acc: {:.0f}%\n'.format(
#     #     100. * src_test_correct / len(src_test_loader.dataset),
#     #     100. * tgt_test_correct / len(tgt_test_loader.dataset)))
#     # return 100. * src_test_correct / len(src_test_loader.dataset), 100. * tgt_test_correct / len(tgt_test_loader.dataset)
#     return src_test_correct / len(d_src_test), tgt_test_correct / len(d_tgt_test)


def test(net, l_src_test, l_tgt_test):
    # discriminator.eval()
    net.E.eval()
    net.C.eval()
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


# def plot_tsne(net, l_src_test, l_tgt_test):
#     tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=600)
#     with torch.no_grad():
#         for i, (data, target) in enumerate(l_src_test):
#             # data, target = data.to(device), target.to(device)
#             data, target = data.cuda(), target.cuda()
#             if net.C.module.use_gumbel:
#                 _, output_logits, output = net.discriminator(data)
#             else:
#                 _, output_logits = net.discriminator(data)
#                 output = F.softmax(output_logits, dim=1)
#             src_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#
#             labels_orig = target.cpu().numpy().tolist()
#             labels_pred = src_test_pred.squeeze().cpu().numpy().tolist()
#             if i == 0:
#                 src_mat = output_logits
#                 src_labels_orig = labels_orig
#                 src_labels_pred = labels_pred
#                 src_label_img = data
#             else:
#                 src_mat = torch.cat((src_mat, output_logits), 0)
#                 src_labels_orig.extend(labels_orig)
#                 src_labels_pred.extend(labels_pred)
#                 src_label_img = torch.cat((src_label_img, data), 0)
#
#
#         src_labels_orig_str = list(map(str, src_labels_orig))
#         src_labels_pred_str = list(map(str, src_labels_pred))
#         src_metadata = list(map(lambda src_labels_orig_str,
#                                        src_labels_pred_str: 'SRC_TST_Ori' + src_labels_orig_str + '_Pre' + src_labels_pred_str,
#                                 src_labels_orig_str, src_labels_pred_str))
#         # label_img.data.mul_(0.5).add_(0.5)
#         # writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=i)
#
#         writer.add_pr_curve('SRC_PR', torch.tensor(src_labels_orig), torch.tensor(src_labels_pred))
#
#         for i, (data, target) in enumerate(l_tgt_test):
#             # data, target = data.to(device), target.to(device)
#             data, target = data.cuda(), target.cuda()
#             if net.C.module.use_gumbel:
#                 _, output_logits, output = net.discriminator(data)
#             else:
#                 _, output_logits = net.discriminator(data)
#                 output = F.softmax(output_logits, dim=1)
#             tgt_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#
#             labels_orig = target.cpu().numpy().tolist()
#             labels_pred = tgt_test_pred.squeeze().cpu().numpy().tolist()
#             if i == 0:
#                 tgt_mat = output_logits
#                 tgt_labels_orig = labels_orig
#                 tgt_labels_pred = labels_pred
#                 tgt_label_img = data
#             else:
#                 tgt_mat = torch.cat((tgt_mat, output_logits), 0)
#                 tgt_labels_orig.extend(labels_orig)
#                 tgt_labels_pred.extend(labels_pred)
#                 tgt_label_img = torch.cat((tgt_label_img, data), 0)
#
#         tgt_labels_orig_str = list(map(str, tgt_labels_orig))
#         tgt_labels_pred_str = list(map(str, tgt_labels_pred))
#         tgt_metadata = list(map(lambda tgt_labels_orig_str,
#                                        tgt_labels_pred_str: 'TGT_TST_Ori' + tgt_labels_orig_str + '_Pre' + tgt_labels_pred_str,
#                                 tgt_labels_orig_str, tgt_labels_pred_str))
#
#         writer.add_pr_curve('TGT_PR', torch.tensor(tgt_labels_orig), torch.tensor(tgt_labels_pred))
#
#         mat = torch.cat((src_mat, tgt_mat), dim=0)
#         metadata = src_metadata + tgt_metadata
#
#         label_img = torch.cat((src_label_img, tgt_label_img), dim=0)
#         label_img.data.mul_(0.5).add_(0.5)
#
#         writer.add_embedding(mat, metadata=metadata, label_img=label_img)
#
#     tsne_results = tsne.fit_transform(mat.cpu().numpy())
#
#     ### Plot TSNE
#     fig = plt.figure()
#     cax = plt.scatter(tsne_results[:len(src_labels_orig), 0], tsne_results[:len(src_labels_orig), 1],
#                       c=torch.tensor(src_labels_orig).numpy(), alpha=1.0,
#                       cmap=plt.cm.get_cmap("jet", d_src_test.n_classes), marker='x',
#                       label='SRC - {}'.format(d_src_test.dataset_name.upper()))
#     cax = plt.scatter(tsne_results[len(src_labels_orig):, 0], tsne_results[len(src_labels_orig):, 1],
#                       c=torch.tensor(tgt_labels_orig).numpy(), alpha=1.0,
#                       cmap=plt.cm.get_cmap("jet", d_tgt_test.n_classes), marker='+',
#                       label='TGT - {}'.format(d_tgt_test.dataset_name.upper()))
#     plt.legend(loc='upper left')
#     # plt.legend(loc='upper right')
#     fig.colorbar(cax, extend='min', ticks=np.arange(0, d_src_test.n_classes))
#     plt.tight_layout()
#     plt.axis('off')
#     # fig.axes.get_xaxis().set_visible(False)
#     # fig.axes.get_yaxis().set_visible(False)
#     writer.add_figure('{}_tsne'.format(args.exp), fig)


# plt.savefig('final_embedding_usps_adapt.png')
def experiment(args):
    exp = args.exp
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    epoch_size = args.epoch_size
    seed = args.seed
    log_file = args.log_file
    workers = args.workers
    image_size = args.image_size
    nc = args.nc
    nz = args.nz
    use_gpu = True
    gpus = args.gpus
    img_dir = args.img_dir
    logs_dir = args.logs_dir
    ckpt_dir = args.ckpt_dir
    plot_interval = args.plot_interval
    experiment = args.exp

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
    use_gen_sqrt = args.use_gen_sqrt  # True/False (False is better : Jobs 7621/7622)
    train_GnE = args.train_GnE  # True/False

    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]

    lambda_MMD = 1.0
    lambda_AE_X = 8.0
    lambda_AE_Y = 8.0
    lambda_rg = 16.0

    lambda_ssl = 1.0
    lambda_sul = 0.0
    lambda_tul = 1.0
    if args.exp in {'usps_mnist', 'mnist_usps', 'svhn_mnist', 'mnist_svhn', 'synsigns_gtsrb', 'syndigits_svhn'}:
        lambda_sal = 0.0
        lambda_tal = 0.0
    elif args.exp in {'cifar_stl', 'stl_cifar'}:
        lambda_sal = 0.0
        lambda_tal = 0.0
    else:
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
    log_file = logs_dir + '/' + hostname + '_' + machinename + '_' + args.exp + '_' + str(log_num) + '_' + args.epoch_size + '_ss_tu' + '.txt'

    # Setup logfile to store output logs
    if log_file is not None:
        while os.path.exists(log_file):
            log_num += 1
            log_file = '{0}/{3}_{1}_{2}_{4}_ss_tu.txt'.format(args.logs_dir, args.exp, log_num, hostname, args.epoch_size)
        # return
    args.log_file = log_file

    args.img_dir = os.path.expanduser(args.img_dir)
    img_dir = os.path.join(absolute_base_path, args.img_dir, hostname + '_' + machinename + '_' + args.exp) + '_' + str(log_num)
    args.img_dir = img_dir

    args.ckpt_dir = os.path.expanduser(args.ckpt_dir)
    ckpt_dir = os.path.join(absolute_base_path, args.ckpt_dir, hostname + '_' + machinename + '_' + args.exp) + '_' + str(log_num) + '_' + args.epoch_size + '_ss_tu'
    args.ckpt_dir = ckpt_dir

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

    try:
        n_iter = 0
        log_str = ''
        log_str += log('\n')
        log_str += log('Output log file {0} created'.format(log_file))
        log_str += log('File used to run the experiment : {0}'.format(absolute_pyfile_path))
        log_str += log('Output image files are stored in {0} directory'.format(img_dir))
        log_str += log('Model files are stored in {0} directory\n'.format(ckpt_dir))

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

            # set current cuda device to 0
            log_str += log('current cuda device = {}'.format(torch.cuda.current_device()))
            torch.cuda.set_device(0)
            log_str += log('using cuda device = {}'.format(torch.cuda.current_device()))
            batch_size *= int(num_gpu)
            # args.buffer_size = batch_size * 10
            # args.learning_rate *= torch.cuda.device_count()

        else:
            raise EnvironmentError("GPU device not available!")

        # if args.nc == 1:
        #     transform = transforms.Compose([
        #         transforms.Resize(args.image_size),
        #         transforms.Grayscale(),
        #         transforms.CenterCrop(args.image_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ])
        # else:
        #     if args.exp in {'usps_mnist', 'mnist_usps'}:
        #         transform = transforms.Compose([
        #             ToRGB(),
        #             transforms.Resize(args.image_size),
        #             transforms.CenterCrop(args.image_size),
        #             transforms.ToTensor(),
        #             transforms.Normalize(
        #                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         ])
        #     else:
        #         transform = transforms.Compose([
        #             transforms.Resize(args.image_size),
        #             transforms.CenterCrop(args.image_size),
        #             transforms.ToTensor(),
        #             transforms.Normalize(
        #                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         ])

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

        if exp == 'syndigits_svhn':
            epoch_size = 'small'

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

        n_train_batches = n_samples // batch_size
        n_src_train_batches = d_src_train.__len__() // batch_size
        n_tgt_train_batches = d_tgt_train.__len__() // batch_size
        n_train_samples = n_train_batches * batch_size
        n_src_train_samples = n_src_train_batches * batch_size
        n_tgt_train_samples = n_tgt_train_batches * batch_size

        pin_memory = True
        if use_sampler:
            try:
                d_src_train_labels = d_src_train.train_labels.numpy()
            except:
                d_src_train_labels = np.concatenate(
                    (d_src_train_inverted.train_labels.numpy(), d_src_train_notinverted.train_labels.numpy()), axis=0)

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
            l_src_train = dutils.DataLoader(d_src_train,
                                            batch_size=batch_size,
                                            sampler=sampler_src,
                                            num_workers=int(workers),
                                            pin_memory=pin_memory,
                                            drop_last=True)

            d_tgt_train_labels = d_tgt_train.train_labels.numpy()
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
            l_tgt_train = dutils.DataLoader(d_tgt_train,
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
            l_src_train = dutils.DataLoader(d_src_train,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=int(workers),
                                            pin_memory=pin_memory,
                                            drop_last=True)

            l_tgt_train = dutils.DataLoader(d_tgt_train,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=int(workers),
                                            pin_memory=pin_memory,
                                            drop_last=True)

        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        # inv_tensor = inv_normalize(tensor)

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
        l_src_test = dutils.DataLoader(d_src_test,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=int(2),
                                       pin_memory=pin_memory,
                                       drop_last=False)

        l_tgt_test = dutils.DataLoader(d_tgt_test,
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

        # construct encoder/decoder modules
        gen_in_dim = nz

        net = Network(args)

        log_str += log('\nBuilding Network from {} ...'.format(args.network_type.upper()))
        log_str += log('Encoder : {}'.format(net.E))
        log_str += log('Classifier : {}'.format(net.C))
        #log_str += log('Generator : {}'.format(net.GS))
        log_str += log('Network Built ...')

        # sigma for MMD
        sigma_list = [sigma / base for sigma in sigma_list]

        # put variable into cuda device
        fixed_noise = Variable(net.fixed_noise, requires_grad=False)

        # For storing the network output of the last buffer_size samples
        args.buffer_size = int(args.buffer_size*n_train_samples)
        p_sum_denominator = torch.rand(args.buffer_size, n_classes).cuda()
        p_sum_denominator /= p_sum_denominator.sum(1).unsqueeze(1).expand_as(p_sum_denominator)
        # p_sum_denominator_src = torch.rand(args.buffer_size, n_classes).cuda()
        # # p_sum_denominator_src = torch.rand(n_tgt_train_samples, n_classes).cuda()
        # p_sum_denominator_src /= p_sum_denominator_src.sum(1).unsqueeze(1).expand_as(p_sum_denominator_src)
        # p_sum_denominator_tgt = p_sum_denominator_src.clone().detach()

        # setup optimizer
        log_str += log('\noptimizerE : {}'.format(net.optimizerE))
        log_str += log('optimizerC : {}'.format(net.optimizerC))
        #log_str += log('optimizerGS : {}'.format(net.optimizerGS))
        #log_str += log('optimizerGT : {}'.format(net.optimizerGT))

        if args.lr_decay_type == 'scheduler':
            scheduler_E = ReduceLROnPlateau(net.optimizerE, 'max')
            scheduler_C = ReduceLROnPlateau(net.optimizerC, 'max')
        #    scheduler_GS = ReduceLROnPlateau(net.optimizerGS, 'max')
        #    scheduler_GT = ReduceLROnPlateau(net.optimizerGT, 'max')

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

        best_src_test_acc = 0  # Best epoch wise src acc
        best_src_test_acc_inter = 0  # Best generator wise src acc
        best_tgt_test_acc = 0  # Best epoch wise src acc
        best_tgt_test_acc_inter = 0  # Best generator wise src acc
        src_test_acc = 0  # dummy init for scheduler

        time = timeit.default_timer()
        total_loss_epochs = []

        #X_src_noise = Variable(torch.FloatTensor(batch_size, gen_in_dim, 1, 1)).normal_(0, 1).cuda()
        #X_tgt_noise = Variable(torch.FloatTensor(batch_size, gen_in_dim, 1, 1)).normal_(0, 1).cuda()

        src_log_prob_buffer = []  # src_log_prob_buffer
        tgt_log_prob_buffer = []  # tgt_log_prob_buffer
        z_src_one_hot_buffer = []  # src_log_prob_buffer
        z_tgt_one_hot_buffer = []  # tgt_log_prob_buffer

        log_str += log('Checkpoint directory to store files for current run : {}'.format(net.args.ckpt_dir))

        net.writer.add_text('Log Text', log_str)

        n_iter = 0
        if args.network_type == 'cdan':
            param_lr = []
            for param_group in net.optimizerE.param_groups:
                param_lr.append(param_group["lr"])
            for param_group in net.optimizerC.param_groups:
                param_lr.append(param_group["lr"])
            schedule_param = {'lr': 0.001, 'gamma': 0.001, 'power': 0.75}  #  optimizer_config["lr_param"]
            # lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
            lr_scheduler = lr_schedule.schedule_dict['inv']

        # # Reset dataloader iterator
        # src_data_iter = iter(l_src_train)
        # tgt_data_iter = iter(l_tgt_train)
        for epoch in range(num_epochs):

            if epoch != 0 and args.lr_decay_type == 'scheduler':
                scheduler_E.step(src_test_acc)
                scheduler_C.step(src_test_acc)
                #scheduler_GS.step(src_test_acc)
                #scheduler_GT.step(src_test_acc)
            elif args.lr_decay_type == 'geometric':
                lr = args.learning_rate * (args.lr_decay_rate ** (epoch // args.lr_decay_period))
                for param_group in net.optimizerC.param_groups:
                    param_group['lr'] = lr
                for param_group in net.optimizerE.param_groups:
                    param_group['lr'] = lr
                # for param_group in net.optimizerGS.param_groups:
                #     param_group['lr'] = lr
                # for param_group in net.optimizerGT.param_groups:
                #     param_group['lr'] = lr
            elif args.lr_decay_type == 'cdan_inv':
                net.optimizerE = lr_scheduler(net.optimizerE, n_iter, **schedule_param)
                net.optimizerC = lr_scheduler(net.optimizerC, n_iter, **schedule_param)
                lr = args.learning_rate
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
            net.writer.add_scalar('weights_loss/unsup/src', lambda_sul, epoch)
            net.writer.add_scalar('weights_loss/unsup/tgt', lambda_tul, epoch)
            net.writer.add_scalar('weights_loss/adv/src', lambda_sal, epoch)
            net.writer.add_scalar('weights_loss/adv/tgt', lambda_tal, epoch)
            net.writer.add_scalar('weights_loss/sup/src_ramp', ramp_sup_weight_in_list[0], epoch)
            net.writer.add_scalar('weights_loss/unsup/src_ramp', ramp_unsup_weight_in_list[0], epoch)
            net.writer.add_scalar('weights_loss/unsup/tgt_ramp', ramp_unsup_weight_in_list[0], epoch)

            net.E.train()
            net.C.train()
            #net.GS.train()
            #net.GT.train()
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
                X_tgt = Variable(X_tgt.cuda())
                X_src_batch_size = X_src.size(0)
                X_tgt_batch_size = X_tgt.size(0)

                # Exit without processing last uneven batch
                if X_src_batch_size != X_tgt_batch_size:
                    break

                # Get only Source labels
                y_src = Variable(y_src.cuda())

                # Train Enc + Dec + Classifier on both Source and Target Domain data
                if net.C.module.use_gumbel:
                    src_enc_out, src_logits_out, z_src_one_hot = net.discriminator(X_src)
                    tgt_enc_out, tgt_logits_out, z_tgt_one_hot = net.discriminator(X_tgt)
                else:
                    src_enc_out, src_logits_out = net.discriminator(X_src)
                    tgt_enc_out, tgt_logits_out = net.discriminator(X_tgt)

                # Supervised classification loss
                src_sup_loss = classification_criterion(src_logits_out, y_src)  # Loss 1 : Supervised loss  # torch.Size([64, 10])

                # # Train Enc + Dec + Classifier on both Source and Target Domain data
                # if net.C.module.use_gumbel:
                #     src_enc_out, src_logits_out, z_src_one_hot = net.discriminator(X_src)
                #     tgt_enc_out, tgt_logits_out, z_tgt_one_hot = net.discriminator(X_tgt)
                # else:
                #     src_enc_out, src_logits_out = net.discriminator(X_src)
                #     tgt_enc_out, tgt_logits_out = net.discriminator(X_tgt)

                # Calculating unsupervised loss and and log loss
                # src_prob = F.softmax(src_logits_out, dim=1)  # torch.Size([64, 10])
                # src_log_prob = F.log_softmax(src_logits_out, dim=1)  # torch.Size([64, 10])
                tgt_prob = F.softmax(tgt_logits_out, dim=1)  # torch.Size([64, 10])
                tgt_log_prob = F.log_softmax(tgt_logits_out, dim=1)  # torch.Size([64, 10])

                if not net.C.module.use_gumbel:
                    with torch.no_grad():
                        # expectation step
                        # src_prob_cp = src_prob.data
                        # tgt_prob_cp = tgt_prob.data
                        # src_prob_cp = src_prob.detach().clone()
                        tgt_prob_cp = tgt_prob.detach().clone()

                        # Update denominator for the unsupervised loss
                        # p_sum_denominator = torch.cat((src_prob_cp, tgt_prob_cp, p_sum_denominator), 0)[0:args.buffer_size]
                        p_sum_denominator = torch.cat((tgt_prob_cp, p_sum_denominator), 0)[0:args.buffer_size]
                        # src_prob_cp *= (prior_src_train / p_sum_denominator.sum(0)).expand_as(src_prob_cp)
                        tgt_prob_cp *= (prior_src_train / p_sum_denominator.sum(0)).expand_as(tgt_prob_cp)
                        # p_sum_denominator_src = torch.cat((src_prob_cp, p_sum_denominator_src), 0)[0:args.buffer_size]
                        # src_prob_cp *= (prior_src / p_sum_denominator_src.sum(0)).expand_as(src_prob_cp)
                        # p_sum_denominator_tgt = torch.cat((tgt_prob_cp, p_sum_denominator_tgt), 0)[0:args.buffer_size]
                        # tgt_prob_cp *= (prior_src / p_sum_denominator_tgt.sum(0)).expand_as(tgt_prob_cp)
                        # p_sum_denominator_src = torch.cat((src_prob_cp, p_sum_denominator_src), 0)[0:n_tgt_train_samples]
                        # src_prob_cp *= (prior_src / p_sum_denominator_src.sum(0)).expand_as(src_prob_cp)
                        # p_sum_denominator_tgt = torch.cat((tgt_prob_cp, p_sum_denominator_tgt), 0)[0:n_tgt_train_samples]
                        # tgt_prob_cp *= (prior_src / p_sum_denominator_tgt.sum(0)).expand_as(tgt_prob_cp)

                        # Find expected labels
                        # _, y_src_pred = src_prob_cp.max(dim=1)
                        # z_src_one_hot = torch.FloatTensor(y_src_pred.size(0), n_classes).cuda()
                        # z_src_one_hot.zero_()
                        # z_src_one_hot.scatter_(1, y_src_pred.unsqueeze(1), 1)
                        # z_src_one_hot = Variable(z_src_one_hot)

                        _, y_tgt_pred = tgt_prob_cp.max(dim=1)
                        z_tgt_one_hot = torch.FloatTensor(X_tgt_batch_size, n_classes).cuda()
                        z_tgt_one_hot.zero_()
                        z_tgt_one_hot.scatter_(1, y_tgt_pred.unsqueeze(1), 1)
                        z_tgt_one_hot = Variable(z_tgt_one_hot)

                # maximization step

                # src_exponent = torch.mm(z_src_one_hot, src_log_prob.t())
                # src_exponent_new = src_exponent - torch.diag(src_exponent).view(X_src_batch_size, 1).expand_as(src_exponent)

                # src_temp = src_exponent_new.exp()
                # src_p_x_z_inv = src_temp.sum(dim=1)
                # src_unsup_loss = src_p_x_z_inv.log().mean()
                # src_unsup_loss = torch.logsumexp(src_exponent_new, dim=1).mean()

                # maximization step
                tgt_exponent = torch.mm(z_tgt_one_hot, tgt_log_prob.t())
                tgt_exponent_new = tgt_exponent - torch.diag(tgt_exponent).view(X_tgt_batch_size, 1).expand_as(tgt_exponent)

                # tgt_temp = tgt_exponent_new.exp()
                # tgt_p_x_z_inv = tgt_temp.sum(dim=1)
                # tgt_unsup_loss = tgt_p_x_z_inv.log().mean()
                tgt_unsup_loss = torch.logsumexp(tgt_exponent_new, dim=1).mean()

                # src_unsup_loss = F.log_softmax(torch.mm(z_src_one_hot, src_log_prob.t()).diag(), dim=0).mean() * net.mone  # torch.Size([64, 10])
                # tgt_unsup_loss = F.log_softmax(torch.mm(z_tgt_one_hot, tgt_log_prob.t()).diag(), dim=0).mean() * net.mone  # torch.Size([64, 10])

                # Adversarial regularization loss
                # with torch.no_grad():
                #     # X_src_noise = torch.FloatTensor(X_src_batch_size, gen_in_dim, 1, 1).normal_(0, 1).cuda()
                #     # X_tgt_noise = torch.FloatTensor(X_tgt_batch_size, gen_in_dim, 1, 1).normal_(0, 1).cuda()
                #     X_src_noise = Variable(X_src_noise.normal_(0, 1))  # total freeze netG_src , netG_tgt
                #     X_tgt_noise = Variable(X_tgt_noise.normal_(0, 1))  # total freeze netG_src , netG_tgt
                #     # X_src_noise.normal_(0, 1)  # total freeze netG_src , netG_tgt
                #     # X_tgt_noise.normal_(0, 1)  # total freeze netG_src , netG_tgt
                #     # X_src_gen = net.GS(X_src_noise).detach()
                #     # X_tgt_gen = net.GT(X_tgt_noise).detach()
                #     X_src_gen = net.GS(X_src_noise)
                #     X_tgt_gen = net.GT(X_tgt_noise)

                # if net.C.module.use_gumbel:
                #     src_gen_enc_out, src_gen_logits_out, _ = net.discriminator(X_src_gen)
                #     tgt_gen_enc_out, tgt_gen_logits_out, _ = net.discriminator(X_tgt_gen)
                # else:
                #     src_gen_enc_out, src_gen_logits_out = net.discriminator(X_src_gen)
                #     tgt_gen_enc_out, tgt_gen_logits_out = net.discriminator(X_tgt_gen)
                #
                # src_adv_loss = F.log_softmax(src_gen_logits_out, dim=1).mean(dim=1).mean() * net.mone
                # tgt_adv_loss = F.log_softmax(tgt_gen_logits_out, dim=1).mean(dim=1).mean() * net.mone

                # lambda_ssl = ramp_sup_weight_in_list[0]
                # lambda_sul = lambda_tul = ramp_unsup_weight_in_list[0]
                total_classifier_loss = lambda_ssl * ramp_sup_weight_in_list[0] * src_sup_loss + lambda_tul * ramp_unsup_weight_in_list[0] * tgt_unsup_loss  # Total Encoder + Classifier loss for Enc + Class training

                ### Add loss and loss weights to tensorboard logs
                net.writer.add_scalar('classifier_loss/sup/src', src_sup_loss.item(), n_iter)
                # net.writer.add_scalar('classifier_loss/unsup/src', src_unsup_loss.item(), n_iter)
                net.writer.add_scalar('classifier_loss/unsup/tgt', tgt_unsup_loss.item(), n_iter)
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

                ### generator traininig starts
                # net.zero_grad()

                # X_src_noise = torch.FloatTensor(X_src_batch_size, gen_in_dim, 1, 1).normal_(0, 1).cuda()
                # X_tgt_noise = torch.FloatTensor(X_tgt_batch_size, gen_in_dim, 1, 1).normal_(0, 1).cuda()
                # X_src_noise = Variable(X_src_noise)  # total freeze netG_src , netG_tgt
                # X_tgt_noise = Variable(X_tgt_noise)  # total freeze netG_src , netG_tgt
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

                # Generator training ends
                if n_iter % plot_interval == 0:
                    # log(
                    #     '    [%3d/%3d][%4d/%4d] (%.2f m): ttl=%.6f tcl=%.6f ssl=%.6f sul=%.6f tul=%.6f sal=%.6f tal=%.6f'
                    #     % (epoch, num_epochs, i, n_train_batches, run_time,
                    #        total_loss.item(), total_classifier_loss.item(), src_sup_loss.item(),
                    #        src_unsup_loss.item(), tgt_unsup_loss.item(),
                    #        src_adv_loss.item(), tgt_adv_loss.item()))

                    # Save image using torch vision
                    with torch.no_grad():
                        fixed_noise = Variable(fixed_noise)  # total freeze netG_src , netG_tgt
                        X_fixed_noise_S = net.GS(fixed_noise).detach()
                        X_fixed_noise_T = net.GT(fixed_noise).detach()

                        image_grid = torch.zeros(
                            (X_fixed_noise_S.shape[0] * 2, X_fixed_noise_S.shape[1], X_fixed_noise_S.shape[2],
                             X_fixed_noise_S.shape[3]),
                            requires_grad=False)
                        # image_grid[0::4, :, :, :] = X_src
                        # image_grid[2::4, :, :, :] = X_tgt
                        image_grid[0::2, :, :, :] = X_fixed_noise_S
                        image_grid[1::2, :, :, :] = X_fixed_noise_T
                        # image_grid = image_grid * std + mean
                        image_grid_n_row = int(torch.sqrt(torch.FloatTensor([X_fixed_noise_S.shape[0]])).item())//2*2
                        image_grid.data.mul_(0.5).add_(0.5)
                        img_grid = vutils.make_grid(image_grid[:image_grid_n_row ** 2, :, :, :], nrow=image_grid_n_row)  # make an square grid of images and plot

                        net.writer.add_image('Generator_images_{0}'.format(args.exp), img_grid, n_iter)  # Tensor
                        # vutils.save_image(img_grid, '{}/{}_train_{:04d}_{:04d}.png'.format(img_dir, args.dataset, epoch + 1, batch_idx + 1))

            # Show the loss
            # log_str += log('Epoch:{:04d}\tLoss:{:6.5f}\n'.format(epoch + 1,  epoch_loss/i))
            total_loss_epochs.append(epoch_loss/(i+1))
            net.writer.add_scalar('classifier_loss/total/epoch', epoch_loss/(i+1), epoch)

            # Create folder to save images
            if epoch == 0:
                # os.system('mkdir -p {0}/Image'.format(args.ckpt_dir))  # create folder to store images
                os.system('mkdir -p {0}/Log'.format(net.args.ckpt_dir))  # create folder to store images

                # vutils.save_image(img_grid, '{}/Image/{}_train_e{:03d}.png'.format(args.ckpt_dir, args.exp, epoch + 1))

            if epoch % 30 == 0:
                # p_sum_denominator.rand_()
                # p_sum_denominator /= p_sum_denominator.sum(1).unsqueeze(1).expand_as(p_sum_denominator)
                # Create folder to save model checkpoints
                net.save_model(suffix='_bkp')
                # os.system('mkdir -p {0}/Log'.format(ckpt_dir))  # create folder to store logs
                image_grid = torch.zeros(
                    (X_src.shape[0] * 2, X_src.shape[1], X_src.shape[2],
                     X_src.shape[3]),
                    requires_grad=False)
                # image_grid[0::4, :, :, :] = X_src
                # image_grid[2::4, :, :, :] = X_tgt
                image_grid[0::2, :, :, :] = X_src
                image_grid[1::2, :, :, :] = X_tgt
                # image_grid = image_grid * std + mean
                image_grid_n_row = int(torch.sqrt(torch.FloatTensor([X_src.shape[0]])).item()) // 2 * 2
                image_grid.data.mul_(0.5).add_(0.5)
                img_grid = vutils.make_grid(image_grid[:image_grid_n_row ** 2, :, :, :],
                                            nrow=image_grid_n_row)  # make an square grid of images and plot

                net.writer.add_image('Original_images_{0}'.format(args.exp), img_grid, epoch)  # Tensor
                # vutils.save_image(img_grid, '{}/{}_train_{:04d}_{:04d}.png'.format(img_dir, args.dataset, epoch + 1, batch_idx + 1))

            # src_train_acc, tgt_train_acc = test(net, d_src_train, d_tgt_train)
            # src_test_acc, tgt_test_acc = test(net, d_src_test, d_tgt_test)
            src_train_acc, tgt_train_acc = test(net, l_src_train, l_tgt_train)
            src_test_acc, tgt_test_acc = test(net, l_src_test, l_tgt_test)

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

                tsne = TSNE(n_components=2, verbose=False, perplexity=50, n_iter=600)
                with torch.no_grad():
                    for j, (data, target) in enumerate(l_src_test):
                        if ((j+1)*l_src_test.batch_size) > args.n_test_samples:
                            break
                        # data, target = data.to(device), target.to(device)
                        data, target = data.cuda(), target.cuda()
                        if net.C.module.use_gumbel:
                            _, output_logits, output = net.discriminator(data)
                        else:
                            _, output_logits = net.discriminator(data)
                            output = F.softmax(output_logits, dim=1)
                        src_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                        labels_orig = target.cpu().numpy().tolist()
                        labels_pred = src_test_pred.squeeze().cpu().numpy().tolist()
                        if j == 0:
                            src_mat = output_logits
                            src_labels_orig = labels_orig
                            src_labels_pred = labels_pred
                            src_label_img = data
                        else:
                            src_mat = torch.cat((src_mat, output_logits), 0)
                            src_labels_orig.extend(labels_orig)
                            src_labels_pred.extend(labels_pred)
                            src_label_img = torch.cat((src_label_img, data), 0)

                    src_labels_orig_str = list(map(str, src_labels_orig))
                    src_labels_pred_str = list(map(str, src_labels_pred))
                    src_metadata = list(map(lambda src_labels_orig_str,
                                                   src_labels_pred_str: 'SRC_TST_Ori' + src_labels_orig_str + '_Pre' + src_labels_pred_str,
                                            src_labels_orig_str, src_labels_pred_str))
                    # label_img.data.mul_(0.5).add_(0.5)
                    # writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=i)

                    net.writer.add_pr_curve('SRC_PR', torch.tensor(src_labels_orig), torch.tensor(src_labels_pred), global_step=epoch)

                    for j, (data, target) in enumerate(l_tgt_test):
                        if ((j+1)*l_src_test.batch_size) > args.n_test_samples:
                            break
                        # data, target = data.to(device), target.to(device)
                        data, target = data.cuda(), target.cuda()
                        if net.C.module.use_gumbel:
                            _, output_logits, output = net.discriminator(data)
                        else:
                            _, output_logits = net.discriminator(data)
                            output = F.softmax(output_logits, dim=1)
                        tgt_test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                        labels_orig = target.cpu().numpy().tolist()
                        labels_pred = tgt_test_pred.squeeze().cpu().numpy().tolist()
                        if j == 0:
                            tgt_mat = output_logits
                            tgt_labels_orig = labels_orig
                            tgt_labels_pred = labels_pred
                            tgt_label_img = data
                        else:
                            tgt_mat = torch.cat((tgt_mat, output_logits), 0)
                            tgt_labels_orig.extend(labels_orig)
                            tgt_labels_pred.extend(labels_pred)
                            tgt_label_img = torch.cat((tgt_label_img, data), 0)

                    tgt_labels_orig_str = list(map(str, tgt_labels_orig))
                    tgt_labels_pred_str = list(map(str, tgt_labels_pred))
                    tgt_metadata = list(map(lambda tgt_labels_orig_str,
                                                   tgt_labels_pred_str: 'TGT_TST_Ori' + tgt_labels_orig_str + '_Pre' + tgt_labels_pred_str,
                                            tgt_labels_orig_str, tgt_labels_pred_str))

                    net.writer.add_pr_curve('TGT_PR', torch.tensor(tgt_labels_orig), torch.tensor(tgt_labels_pred), global_step=epoch)

                    mat = torch.cat((src_mat, tgt_mat), dim=0)
                    metadata = src_metadata + tgt_metadata

                    label_img = torch.cat((src_label_img, tgt_label_img), dim=0)
                    label_img.data.mul_(0.5).add_(0.5)

                    # net.writer.add_embedding(mat, metadata=metadata, label_img=label_img, n_iter=(epoch * len(tgt_labels_pred)) + j)
                    net.writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=epoch)

                tsne_results = tsne.fit_transform(mat.cpu().numpy())

                ### Plot TSNE

                c = torch.tensor(src_labels_orig).numpy()
                if args.exp in {'cifar_stl', 'st_cifar'}:
                    c = ['airplane', 'automobile',
                         'bird', 'cat',
                         'deer', 'dog',
                         'horse',
                         'ship', 'truck']  # 'frog',
                elif args.exp in {'synsigns_gtsrb'}:
                    c = d_tgt_train.signnames_list
                else:
                    c = torch.tensor(src_labels_orig).numpy()

                fig = plt.figure()
                cax = plt.scatter(tsne_results[:len(src_labels_orig), 0], tsne_results[:len(src_labels_orig), 1],
                                  c=torch.tensor(src_labels_orig).numpy(), alpha=1.0,
                                  cmap=plt.cm.get_cmap("jet", d_src_test.n_classes), marker='*',  # 'x',
                                  label='SRC - {} - #{}'.format(d_src_test.dataset_name.upper(), len(src_labels_orig)))
                cax = plt.scatter(tsne_results[len(src_labels_orig):, 0], tsne_results[len(src_labels_orig):, 1],
                                  c=torch.tensor(tgt_labels_orig).numpy(), alpha=1.0,
                                  cmap=plt.cm.get_cmap("jet", d_tgt_test.n_classes), marker='+',  # '+',
                                  label='TGT - {} - #{}'.format(d_tgt_test.dataset_name.upper(), len(tgt_labels_orig)))
                plt.legend(loc='upper left')
                # plt.legend(loc='upper right')
                fig.colorbar(cax, extend='min', ticks=np.arange(0, d_src_test.n_classes))
                # plt.text(0, 0.1, r'$\delta$',
                #          {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
                #           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
                # plt.figtext(.5, .9, '{} : ACCURACIES\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(args.exp.upper(), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=10)
                # plt.tight_layout()
                plt.axis('off')
                os.system('mkdir -p {}/plots'.format(net.args.ckpt_dir))
                plt.savefig('{}/plots/{}_tsne_epoch{:03d}.png'.format(net.args.ckpt_dir, net.args.exp, epoch))
                plt.title('{} : ACCURACIES\nS_TRN= {:5.2%}, T_TRN= {:5.2%}\nS_TST= {:5.2%}, T_TST= {:5.2%}'.format(args.exp.upper(), src_train_acc/(l_src_train.__len__() * batch_size), tgt_train_acc/(l_tgt_train.__len__() * batch_size), src_test_acc/len(d_src_test), tgt_test_acc/len(d_tgt_test)), fontsize=8)
                plt.savefig('{}/plots/{}_tsne_epoch{:03d}_acc.png'.format(net.args.ckpt_dir, net.args.exp, epoch))
                # net.writer.add_figure('{}_tsne'.format(args.exp), fig, global_step=(epoch * len(tgt_labels_pred)) + j)
                plt.tight_layout()
                net.writer.add_figure('{}_tsne'.format(args.exp), fig, global_step=epoch)

            else:
                # log_str += log('    Epoch:{:3d}/{:3d} #batch:{:3d} took {:2.2f}m - Loss:{:6.5f} SRC TRAIN ACC = {:2.3%}, TGT TRAIN ACC = {:2.3%} SRC TEST ACC = {:2.3%}, TGT TEST ACC = {:2.3%}'.format(
                #         epoch, num_epochs, i, run_time, epoch_loss/i, src_train_acc, tgt_train_acc, src_test_acc, tgt_test_acc))
                tgt_prefix = '  '

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
                                              'lsun', 'imagenet',
                                              'folder', 'lfw'
                                              ], default='mnist',  help='mnist | cifar10 | cifar100 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', type=str, default='./../sesml/data', help='path to args.dataset')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate (Adam)')
    parser.add_argument('--num_epochs', type=int, default=3000, help='number of epochs : (default=3000)')
    parser.add_argument('--ramp', type=int, default=0, help='ramp for epochs')
    # parser.add_argument('--buffer_size', type=int, default=10000, help='length of the buffer for latent feature selection : (default=10000)')
    parser.add_argument('--buffer_size', type=float, default=0.4, help='length of the buffer for latent feature selection : (default=10000)')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size : (default=64)')
    parser.add_argument('--seed', type=int, default=1126, help='random seed (0 for time-based)')
    parser.add_argument('--log_file', type=str, default='', help='log file path (none to disable)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
    parser.add_argument('--image_size', type=int, default=28, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='number of channel')
    parser.add_argument('--nz', type=int, default=100, help='dimension of noise input to generator')
    parser.add_argument('--gpus', type=str, default='0,1', help='using gpu device id')
    # parser.add_argument('--gpus', type=str, default='3,2,1,0', help='using gpu device id')
    parser.add_argument('--plot_interval', type=int, default=50, help='Number of plots required to save every iteration')
    parser.add_argument('--img_dir', type=str, default='./images', help='path to save images')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='path to save logs')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint', help='path to save model checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default='', help='path to checkpoint')

    # Hyperparameter search for reproducing - hp448.sh
    parser.add_argument('--use_sampler', action='store_true', default=True, help='use sampler for dataloader (default : False)')
    parser.add_argument('--use_gen_sqrt', action='store_true', default=False, help='use squareroot for MMD loss (default : False)')
    parser.add_argument('--train_GnE', action='store_true', default=False, help='train encoder with generator training (default : False)')
    parser.add_argument('--network_type', type=str, default='se', help='type of network [se|mcd|dade|cdan]')
    parser.add_argument('--use_tied_gen', action='store_true', default=False, help='use a single generator for source and target domain (default : False)')
    parser.add_argument('--epoch_size', default='large', choices=['large', 'small', 'source', 'target'], help='epoch size is either that of the smallest dataset, the largest, the source, or the target (default=source)')
    parser.add_argument('--use_drop', action='store_true', default=True, help='use dropout for classifier (default : False)')
    parser.add_argument('--use_bn', action='store_true', default=True, help='use batchnorm for classifier (default : False)')
    parser.add_argument('--lr_decay_type', choices=['scheduler', 'geometric', 'none', 'cdan_inv'], default='geometric', help='lr_decay_type (default=none)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.6318, help='learning rate (Adam)')
    parser.add_argument('--lr_decay_period', type=float, default=30, help='learning rate (Adam)')
    parser.add_argument('--weight_init', type=str, default='none', help='type of weight initialization')
    parser.add_argument('--use_gumbel', action='store_true', default=False, help='use Gumbel softmax for label selection (default : False)')
    parser.add_argument('--n_test_samples', type=int, default=1000, help='number of test samples used for plotting t-SNE')

    #     parser = get_dade_args(parser)
    args = parser.parse_args()
    print(args)

    experiment(args)
    '''
    single generator model with combined training
    SE network
    '''
