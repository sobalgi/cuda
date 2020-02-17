from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import numpy as np
import pickle
import torch


class BOOKS(data.Dataset):
    n_classes = 2
    domain = 'books'

    def __init__(self, root_file, labeled=True, transform=None, target_transform=None, feature_num=5000):
        self.root_file = os.path.expanduser(root_file)
        self.transform = transform
        self.target_transform = target_transform
        self.labeled = labeled  # training set or test set

        # print(f'Loading mSDA Preprocessed Multi-Domain Amazon data for {self.domain} Domain')
        dataset = pickle.load(open(root_file, 'rb'))[self.domain]

        if labeled:
            self.split = 'labeled'
        else:
            self.split = 'unlabeled'

        lx, ly = dataset[self.split]
        # self.lx = lx
        # self.ly = ly

        # self.data = lx
        self.data = torch.from_numpy(lx.toarray()).float()
        if feature_num > 0:
            self.data = self.data[:, : feature_num]

        self.loc = self.data.mean(dim=0)
        self.scale = self.data.std(dim=0)
        self.covariance_matrix = torch.diag(self.scale)

        # self.labels = ly
        self.labels = torch.from_numpy(ly).long()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (feature, target) where target is index of the target class.
        """
        feature, target = self.data[index], self.labels[index]

        if self.transform is not None:
            feature = self.transform(feature)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feature, target

    def __len__(self):
        return self.labels.shape[0]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root_file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
