import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import tifffile
from PIL import Image


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 改---数据增强
        # if random.random() > 0.5:
        # image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        # image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        sample = {'image': image, 'label': label}
        # sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, label_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = os.listdir(base_dir)
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.label_dir = base_dir + "_label/"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name)
            # label_path = "/home/weizhihao/project_TransUNet_111/data/Synapse/train_npz_label/" + slice_name
            # label_path = "/home/weizhihao/project_TransUNet_111/data/Synapse/train_npz_label/" + slice_name[
            #                                                                                      :-7] + "fluo.png"
            label_path = self.label_dir + slice_name[:-7] + "fluo.png"

            # with  tifffile.TiffFile(data_path) as reader:
            #     image = reader.asarray()
            #     with tifffile.TiffFile(label_path) as reader1:
            #         label = reader1.asarray()

            # image = image / 65535
            # label = label / 65535

            image = Image.open(data_path)
            image = np.array(image) / 255
            label = Image.open(label_path)
            label = np.array(label) / 255

            # with  tifffile.TiffFile(data_path) as reader:
            #     # print(type(reader))
            #     image = reader.asarray()
            #     # print(type(tif_np))
            #     with tifffile.TiffFile(label_path) as reader1:
            #         label = reader1.asarray()
            #
            # #data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # #data = np.load(data_path)
            # #image, label = data['image'], data['label']
            # image=image/65525
            # label=label/65525
        else:
            vol_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, vol_name)
            label_path = "/home/weizhihao/project_TransUNet_1/data/Synapse/train_npz_label/" + vol_name[
                                                                                               :-7] + "fluo.tif"
            with  tifffile.TiffFile(data_path) as reader:
                # print(type(reader))
                image = reader.asarray()
                # print(type(tif_np))
                with tifffile.TiffFile(label_path) as reader1:
                    label = reader1.asarray()

            image = image / 65525
            label = label / 65525

            # filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            # data = h5py.File(filepath)
            # image, label = data['image'][:], data['label'][:]
        # print(image.shape)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
