import os
import random
import numpy
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import imageio
import tifffile
from libtiff import TIFF
import torchvision.transforms as transforms
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
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([796], [751])
        ])
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([579], [1053])
        ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        # print("before image",image.shape)
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (1120 / x, 1120 / y), order=3)  # why not 3?
        #     label = zoom(label, (1120 / x, 1120 / y), order=0)
        # 1120 -》 224
        # 切割(密集滑动窗口、随机切割) or  缩放x
        # 图片切割
        side = 224
        num_h, num_w = 1120 // side, 1120 // side
        h = np.random.randint(0, num_h)
        w = np.random.randint(0, num_w)
        # side = 512
        # num_h, num_w = 2048 // side, 2048 // side
        # h = np.random.randint(0, num_h)
        # w = np.random.randint(0, num_w)
        image = image[h * side:(h + 1) * side, w * side:(w + 1) * side]
        label = label[h * side:(h + 1) * side, w * side:(w + 1) * side]

        # image=np.expand_dims(image,0)
        # label=np.expand_dims(label,0)
        # print("after image", image.shape)

        # 实现z-score标准化 Standardization(缩放到均值为0，方差为1)
        # std_1 = np.std(image)
        # std_2 = np.std(label)
        # mean_1 = np.mean(image)
        # mean_2 = np.mean(label)
        #
        # image = (image - mean_1) / std_1
        # label = (label - mean_2) / std_2

        # 实现归一化 (缩放到0和1之间，保留原始数据的分布)
        # max_image = image.max()
        # min_image = image.min()
        # image = image - min_image
        # image = image / (max_image - min_image)
        #
        # max_label = label.max()
        # min_label = label.min()
        # label = label - min_label
        # label = label / (max_label - min_label)

        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        image = self.transform_img(image.astype(np.float32))
        print('image shape ',image.shape)
        label = self.transform_label(label.astype(np.float32))
        sample = {'image': image, 'label': label}

        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, label_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = os.listdir(base_dir)
        self.label_dir = label_dir
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name)
            label_name = slice_name.replace("Retardance", "568")
            label_path = os.path.join(self.label_dir, label_name)

            image = TIFF.open(data_path, mode='r').read_image()
            # print("read", image.shape)
            label = TIFF.open(label_path, mode='r').read_image()
            # with tifffile.TiffFile(data_path) as reader:
            #     image=reader.asarray()
            #     print("read",image.shape)
            #     with tifffile.TiffFile(label_path) as reader1:
            #         label = reader1.asarray()


        else:
            vol_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, vol_name)
            label_name = vol_name.replace("Retardance", "568")
            label_path = os.path.join(self.label_dir, label_name)
            with  tifffile.TiffFile(data_path) as reader:
                # print(type(reader))
                image = reader.asarray()
                # print(type(tif_np))
                with tifffile.TiffFile(label_path) as reader1:
                    label = reader1.asarray()

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
