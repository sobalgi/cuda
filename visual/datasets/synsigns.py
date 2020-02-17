import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import tqdm
import torch


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class SYNSIGNS(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """
    dataset_name = 'synsigns'
    processed_folder = 'processed'
    n_classes = 43

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.loader = default_loader

        dset_path = os.path.join(self.root, self.dataset_name, self.processed_folder)
        labels_path = os.path.join(dset_path, 'train_labelling.txt')

        if not os.path.exists(labels_path):
            print('Labels path {} does not exist'.format(labels_path))
            sys.exit(0)

        # Open the file that lists the image files along with their ground truth class
        lines = [line.strip() for line in open(labels_path, 'r').readlines()]
        lines = [line for line in lines if line != '']

        images = []
        labels = []

        for line in tqdm.tqdm(lines):
            image_filename, label, _ = line.split()
            image_path = os.path.join(dset_path, image_filename)

            if not os.path.exists(image_path):
                print('Could not find image file {} mentioned in annotations'.format(image_path))
                return

            images.append(image_path)
            labels.append(int(label))

        self.images = images
        if self.train:
            self.train_labels = labels
            self.train_labels = torch.tensor(self.train_labels).type(torch.LongTensor)
        else:
            self.test_labels = labels
            self.test_labels = torch.tensor(self.test_labels).type(torch.LongTensor)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_path = self.images[index]
        if self.train:
            target = self.train_labels[index]
        else:
            target = self.test_labels[index]

        sample = self.loader(image_path)
        # sample = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
