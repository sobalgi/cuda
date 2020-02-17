#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import datasets as mydset
import pickle
import numpy as np

from os import listdir
from os.path import join
import random


def get_data(args, domain='books'):
    print(f'Loading mSDA Preprocessed Multi-Domain Amazon data for {domain} Domain')
    if domain == 'books':
        labeled_dataset = mydset.BOOKS(root_file=args.dataroot, labeled=True)
        unlabeled_dataset = mydset.BOOKS(root_file=args.dataroot, labeled=False)

    elif domain == 'dvd':
        labeled_dataset = mydset.DVD(root_file=args.dataroot, labeled=True)
        unlabeled_dataset = mydset.DVD(root_file=args.dataroot, labeled=False)

    elif domain == 'electronics':
        labeled_dataset = mydset.ELECTRONICS(root_file=args.dataroot, labeled=True)
        unlabeled_dataset = mydset.ELECTRONICS(root_file=args.dataroot, labeled=False)

    elif domain == 'kitchen':
        labeled_dataset = mydset.KITCHEN(root_file=args.dataroot, labeled=True)
        unlabeled_dataset = mydset.KITCHEN(root_file=args.dataroot, labeled=False)
    else:
        raise ValueError("Unknown dataset %s" % (domain))

    print('{}: labeled: count={}, unlabeled: count={}'.format(domain.upper(), labeled_dataset.__len__(), unlabeled_dataset.__len__()))

    return labeled_dataset, unlabeled_dataset
