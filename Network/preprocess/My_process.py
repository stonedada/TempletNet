import numpy as np
import cv2
import os

import tifffile

root = '/home/stone/dataset/'  # change to your data folder path
data_f = ['tif/train/', 'tif/val/', 'tif/test/']
mask_f = ['tif/train_label/', 'tif/val_label/',
          'tif/test_label/']
set_size = [128, 32, 32]
save_name = ['train', 'val', 'test']

height = 192
width = 256

for j in range(3):

    print('processing ' + data_f[j] + '......')

    count_im = 0
    count_gt = 0
    number_of_height = int(2048/height)
    number_of_width = int(2048 / width)
    length = set_size[j] * number_of_height*number_of_width
    imgs = np.uint8(np.zeros([length, height, width]))
    masks = np.uint8(np.zeros([length, height, width]))

    path = root + data_f[j]
    mask_p = root + mask_f[j]
    for i in os.listdir(path):
        img = tifffile.imread(path + i)
        m_path = mask_p + i.replace("Retardance", "568")
        mask = tifffile.imread(m_path)
        for cnt_row in range(number_of_height):  # 行
            for cnt_column in range(number_of_width):  # 列
                h = cnt_row * height
                w = cnt_column * width
                crop_img = img[h: height + h, w: w + width]
                imgs[count_im + 1] = crop_img

                crop_mask = mask[h: height + h, w: w + width]
                masks[count_gt + 1] = crop_mask

    out_dir = root + 'npy_256'
    np.save('{}/data_{}.npy'.format(out_dir, save_name[j]), imgs)
    np.save('{}/mask_{}.npy'.format(out_dir, save_name[j]), masks)
