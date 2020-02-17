import torch.utils.data as data

from PIL import Image

import os
import os.path
import pandas as pd
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


class GTSRB(data.Dataset):
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
    dataset_name = 'gtsrb'
    processed_folder = 'processed'
    n_classes = 43

    def __init__(self, root, train=True, transform=None, target_transform=None, ignore_roi=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.loader = default_loader
        self.ignore_roi = ignore_roi

        images = []
        roi_X1 = []
        roi_X2 = []
        roi_Y1 = []
        roi_Y2 = []
        labels = []

        signnames_path = os.path.join(self.root, self.dataset_name, self.processed_folder, 'signnames.csv')
        signnames = pd.read_csv(signnames_path)
        self.signnames_list = list(signnames['SignName'])

        if train:
            dset_path = os.path.join(self.root, self.dataset_name, self.processed_folder, 'Final_Training', 'Images')

            # for clf_dir_name in os.listdir(dset_path):
            for clf_dir_name in tqdm.tqdm(os.listdir(dset_path), desc='Class'):
                clf_ndx = int(clf_dir_name)
                clf_path = os.path.join(dset_path, clf_dir_name)

                anno_path = os.path.join(clf_path, 'GT-{:05d}.csv'.format(clf_ndx))

                if not os.path.exists(anno_path):
                    print('ERROR!!! Could not find annotations file {}'.format(anno_path))
                    return False

                annotations = pd.read_csv(anno_path, sep=';')

                for index, row in annotations.iterrows():
                    # for index, row in tqdm.tqdm(annotations.iterrows(), desc='Images', total=len(annotations.index)):
                    image_filename = row['Filename']
                    image_path = os.path.join(clf_path, image_filename)

                    if not os.path.exists(image_path):
                        print('ERROR!!!  Could not find image file {} mentioned in annotations'.format(image_path))
                        return False

                    images.append(image_path)

                    # Crop out the region of interest
                    roi_x1 = int(row['Roi.X1'])
                    roi_x2 = int(row['Roi.X2'])
                    roi_y1 = int(row['Roi.Y1'])
                    roi_y2 = int(row['Roi.Y2'])
                    roi_X1.append(roi_x1)
                    roi_X2.append(roi_x2)
                    roi_Y1.append(roi_y1)
                    roi_Y2.append(roi_y2)

                    class_id = int(row['ClassId'])
                    labels.append(class_id)
        else:
            dset_path = os.path.join(self.root, self.dataset_name, self.processed_folder, 'Final_Test', 'Images')
            anno_path = os.path.join(dset_path, 'GT-final_test.csv')

            if not os.path.exists(anno_path):
                print('ERROR!!! Could not find annotations file {}'.format(anno_path))
                return False

            annotations = pd.read_csv(anno_path, sep=';')

            for index, row in annotations.iterrows():
                # for index, row in tqdm.tqdm(annotations.iterrows(), desc='Images', total=len(annotations.index)):
                image_filename = row['Filename']
                image_path = os.path.join(dset_path, image_filename)

                if not os.path.exists(image_path):
                    print('ERROR!!!  Could not find image file {} mentioned in annotations'.format(image_path))
                    return False

                images.append(image_path)

                # Crop out the region of interest
                roi_x1 = int(row['Roi.X1'])
                roi_x2 = int(row['Roi.X2'])
                roi_y1 = int(row['Roi.Y1'])
                roi_y2 = int(row['Roi.Y2'])
                roi_X1.append(roi_x1)
                roi_X2.append(roi_x2)
                roi_Y1.append(roi_y1)
                roi_Y2.append(roi_y2)

                class_id = int(row['ClassId'])
                labels.append(class_id)

        self.images = images
        self.roi_X1 = roi_X1
        self.roi_X2 = roi_X2
        self.roi_Y1 = roi_Y1
        self.roi_Y2 = roi_Y2
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

        w, h = sample.size
        if not self.ignore_roi:
            upper = self.roi_Y2[index]
            left = self.roi_X1[index]
            lower = self.roi_Y1[index]
            right = self.roi_X2[index]

            if right < left:
                left = self.roi_X2[index]
                right = self.roi_X1[index]
            if upper < lower:
                lower = self.roi_Y2[index]
                upper = self.roi_Y1[index]

            sample = sample.crop((left, h-upper, right, h-lower))
            # image_data = image_data[roi_y1:roi_y2, roi_x1:roi_x2, :]

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
