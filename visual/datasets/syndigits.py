from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from torchvision.datasets.utils import download_url, check_integrity


class SYNDIGITS(data.Dataset):
    """`SYNDIGITS <https://doc-04-4o-docs.googleusercontent.com/docs/securesc/9jndu35sq8t97c9r6s1fimdpdlh6qr92/momj6kiefs5an2sholhbv54jco65qibd/1543924800000/02005382952228186512/16112563594121367495/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU?e=download&nonce=al85v5nr3js7g&user=16112563594121367495&hash=gi08apubdppjqneokhirdds96mlpgj1f>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'https://doc-04-4o-docs.googleusercontent.com/docs/securesc/9jndu35sq8t97c9r6s1fimdpdlh6qr92/momj6kiefs5an2sholhbv54jco65qibd/1543924800000/02005382952228186512/16112563594121367495/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU?e=download&nonce=al85v5nr3js7g&user=16112563594121367495&hash=gi08apubdppjqneokhirdds96mlpgj1f',
    ]

    raw_folder = 'raw'
    dataset_name = 'syndigits'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    n_classes = 10

    file_md5 = ""

    split_list = {
        'train': ["synth_train_32x32.mat"],
        'test': ["synth_test_32x32.mat"]}

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.download = download  # Auto Download dataset not implemented yet. Only manual download
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if train:
            split = 'train'  # training set or test set or extra set
        else:
            split = 'test'  # training set or test set or extra set

        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.filename = self.split_list[split][0]

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.dataset_name, self.processed_folder, self.filename))

        # mat = loadmat(mat_path)
        # m_X = mat['X'].astype(np.uint8).transpose(3, 2, 0, 1)
        # m_y = mat['y'].astype(np.int32)[:, 0]
        # m_y[m_y == 10] = 0

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        if train:
            self.train_labels = loaded_mat['y'].astype(np.int64).squeeze()
            np.place(self.train_labels, self.train_labels == 10, 0)
            self.train_labels = torch.from_numpy(self.train_labels).type(torch.LongTensor)
        else:
            self.test_labels = loaded_mat['y'].astype(np.int64).squeeze()
            np.place(self.test_labels, self.test_labels == 10, 0)
            self.test_labels = torch.from_numpy(self.test_labels).type(torch.LongTensor)
        # self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], int(self.train_labels[index])
        else:
            img, target = self.data[index], int(self.test_labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
