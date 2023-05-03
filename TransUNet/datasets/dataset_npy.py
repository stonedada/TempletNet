import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import tifffile
from PIL import Image
from torchvision.transforms import transforms




class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.img_transform = transforms.Compose([
            transforms.Resize(self.output_size)
        ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image = self.img_transform(image)
        label = self.img_transform(label)
        sample = {'image': image, 'label': label}

        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, label_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = os.listdir(base_dir)
        self.data_dir = base_dir
        self.label_dir = base_dir + "_label/"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name)
            label_name = slice_name.replace('c001', 'c000').replace('sl0-3', 'sl0-1')
            label_path = self.label_dir + label_name
            image = np.load(data_path)
            label = np.load(label_path)

        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name)
            label_name = slice_name.replace('c001', 'c000')
            label_path = self.label_dir + label_name
            image = np.load(data_path)
            label = np.load(label_path)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].split('.')[0]
        return sample
