from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
from .cifar9 import CIFAR9
from torchvision.datasets.utils import download_url, check_integrity
import torch


class STL9(data.Dataset):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
    dataset_name = "stl9"
    processed_folder = 'processed'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')
    n_classes = 9

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        # self.root = os.path.join(self.root, self.dataset_name, self.processed_folder)
        self.transform = transform
        self.target_transform = target_transform

        if train:
            split = 'train'  # training set or test set or extra set
        else:
            split = 'test'  # training set or test set or extra set

        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))

        self.split = split  # train/test/unlabeled set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. '
                'You can use download=True to download it')

        # now load the picked numpy arrays
        if self.split == 'train':
            self.data, self.train_labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
        elif self.split == 'train+unlabeled':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.test_labels = self.__loadfile(
                self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(
            self.root, self.dataset_name, self.processed_folder, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

        # Remap class indices to match CIFAR-9:
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
        if self.split == 'train':
            self.train_labels = cls_mapping[self.train_labels]

            # Remove all samples from skipped classes
            label_mask = self.train_labels != -1
            self.train_labels = list(self.train_labels[label_mask])
            self.train_labels = torch.tensor(self.train_labels).type(torch.LongTensor)
        elif self.split == 'train+unlabeled':
            raise NotImplementedError
        elif self.split == 'unlabeled':
            raise NotImplementedError
        else:  # self.split == 'test':
            self.test_labels = cls_mapping[self.test_labels]

            # Remove all samples from skipped classes
            label_mask = self.test_labels != -1
            self.test_labels = list(self.test_labels[label_mask])
            self.test_labels = torch.tensor(self.test_labels).type(torch.LongTensor)

        self.data = self.data[label_mask]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'train':
            if self.train_labels is not None:
                img, target = self.data[index], int(self.train_labels[index])
            else:
                img, target = self.data[index], None
        elif self.split == 'train+unlabeled':
            raise NotImplementedError
        elif self.split == 'unlabeled':
            raise NotImplementedError
        else:  # self.split == 'test':
            if self.test_labels is not None:
                img, target = self.data[index], int(self.test_labels[index])
            else:
                img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.dataset_name, self.processed_folder, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.dataset_name, self.processed_folder, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self):
        root = os.path.join(self.root, self.dataset_name, self.processed_folder)
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = os.path.join(self.root, self.dataset_name, self.processed_folder)
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

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
