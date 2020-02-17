from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch


class USPS(data.Dataset):
    """USPS <'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'> handwritten digits.
    Homepage: http://statweb.stanford.edu/~hastie/ElemStatLearn/data.html
    Images are 16x16 grayscale images in the range [0, 1].

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
        'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz',
        'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz'
    ]

    raw_folder = 'raw'
    dataset_name = 'usps'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    n_classes = 10

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.dataset_name, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.dataset_name, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.dataset_name, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.dataset_name, self.processed_folder, self.test_file))

    def download(self):
        """Download the USPS data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.dataset_name, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.dataset_name, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.dataset_name, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_data_file(os.path.join(self.root, self.dataset_name, self.raw_folder, 'zip.train'))

        )
        test_set = (
            read_data_file(os.path.join(self.root, self.dataset_name, self.raw_folder, 'zip.test'))

        )
        with open(os.path.join(self.root, self.dataset_name, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.dataset_name, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def read_data_file(path):
    images = []
    labels = []
    with open(path, 'r') as f:
        for line in f.readlines():
            sample = line.strip().split()
            labels.append(int(float(sample[0])))
            flat_img = [float(val) for val in sample[1:]]
            flat_img = np.array(flat_img, dtype=np.float32)
            images.append(flat_img.reshape((1, 16, 16)))

    labels = np.array(labels).astype(np.int32)
    images = np.concatenate(images, axis=0)
    images = (images * 0.5 + 0.5) * 255  # Scale from [-1, 1] range to [0, 1]
    images = images.astype(np.uint8)

        # df = pd.read_csv(path, delimiter='\s', header=None)
        # length = df.shape[0]
        # values = df.as_matrix()
        # labels = torch.from_numpy(values[:, 0]).type(torch.LongTensor)
        # images = values[:, 1:].transpose((0, 2, 3, 1))  # convert to HWC
        # images = torch.from_numpy(images)

    return torch.from_numpy(images).type(torch.ByteTensor), torch.from_numpy(labels).type(torch.LongTensor)
