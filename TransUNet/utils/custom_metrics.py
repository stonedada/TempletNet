# -*- coding : utf-8 -*-
# @Author   :   stone
# @Github   :   https://github.com/stonedada

import math
import torch
import numpy as np
from scipy.stats._stats_py import _sum_of_squares


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice

def mean_dice(y_true, y_pred, thresh):
    """
    compute mean dice for binary regression task via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    a = np.abs(y_pred - y_true)
    b = (a < thresh).astype(int)
    intersection = np.sum(b, axis=axes)
    mask_sum = y_true.shape[0]*y_true.shape[1]*2
    
    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou

def mean_iou(y_true, y_pred, thresh):
    """
    compute mean iou for regression task via numpy
    """
    axes = (0, 1)
    a = np.abs(y_pred - y_true)
    b = (a < thresh).astype(int)
    intersection = np.sum(b, axis=axes)
    mask_sum = y_true.shape[0]*y_true.shape[1]*2
    union = mask_sum - intersection
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou

# 计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

# 计算Pearson系数
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)   # 计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p
def pearsonr(x, y):

    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(_sum_of_squares(xm) * _sum_of_squares(ym))
    r = r_num / r_den

    return  r

def r2_metric(target, prediction):
    """Coefficient of determination of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float coefficient of determination
    """
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    cur_r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return cur_r2

def tensor_to_tif(input: torch.Tensor, channel: int = 32, shape: tuple = (256, 256), save_path: str = "/home/yingmuzhi/microDL_3D/_yingmuzhi/output_tiff.tiff", dtype: str="uint16", mean=0, std=1):
    """
    introduce:
        transform tensor to tif and save
    """
    import tifffile
    
    npy = input.numpy()
    npy = npy * std + mean
    npy = npy.astype(dtype=dtype)
    npy = npy.reshape((1, channel, shape[0], shape[1]))
    tifffile.imsave(save_path, npy)
    print("done")