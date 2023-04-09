# -*- coding : utf-8 -*-
# @Author   :   stone
# @Github   :   https://github.com/stonedada
import cv2
import numpy as np
import os


def img_cut(img, side, img_save_path, num):
    h_size, w_size = img.shape[0], img.shape[1]  # 高宽尺寸
    # 共截取（num_h+1）* （num_w+1）张图片，
    num_h, num_w = h_size // side, w_size // side
    # 截取后图片的计数编号
    num = (num_h + 1) * (num_w + 1) * num

    img = np.array(img)
    img_crop = np.zeros((side, side, 3))
    image = []
    for h in range(0, num_h):
        for w in range(0, num_w):
            img_crop = img[h * side:(h + 1) * side, w * side:(w + 1) * side]
            image.append(img_crop)
            if w == num_w - 1:
                img_crop = img[h * side:(h + 1) * side, w_size - side:w_size]
                image.append(img_crop)
        if h == num_h - 1:
            for w in range(0, num_w):
                img_crop = img[h_size - side:h_size, w * side:(w + 1) * side]
                image.append(img_crop)
                if w == num_w - 1:
                    img_crop = img[h_size - side:h_size, w_size - side:w_size]
                    image.append(img_crop)

    for i in range(0, len(image)):
        image_i = image[i]
        path_image_i = img_save_path + str(num + i + 1) + str('.jpg')
        cv2.imwrite(path_image_i, image_i)


if __name__ == '__main__':
    img_load_path = './test/img/'  # 原图片路径
    img_save_path = './test/img_change/'  # 截取后保存路径
    side = 1000  # 裁剪大小1000*1000
    img_names = os.listdir(img_load_path)
    i = 0  # 来记录第几张图片
    for img_name in img_names:
        img_path = os.path.join(img_load_path, img_name)
        img = cv2.imread(img_path)
        img_cut(img, side=side, img_save_path=img_save_path, num=i)
        i += 1