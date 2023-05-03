# -*- coding : utf-8 -*-
# @Author   :   stone
# @Github   :   https://github.com/stonedada

import math
import sklearn
import numpy as np
from scipy.stats._stats_py import _sum_of_squares
from sklearn.metrics import f1_score

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
    mask_sum = y_true.shape[0] * y_true.shape[1] * 2

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
    mask_sum = y_true.shape[0] * y_true.shape[1] * 2
    union = mask_sum - intersection
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


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

    return r


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

def dice_metric(target, prediction):
    """Dice similarity coefficient (F1 score) of binary target and prediction.
    Reports global metric.
    Not using mask decorator for binary data evaluation.

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float dice: Dice for binarized data
    """
    target_bin = binarize_array(target)
    pred_bin = binarize_array(prediction)
    return f1_score(target_bin, pred_bin, average='micro')


def iou_metric(target, prediction):
    """
    compute mean iou for binary segmentation map via numpy
    """
    y_pred = binarize_array(prediction)
    y_true = binarize_array(target)
    intersection = np.sum(np.abs(y_pred * y_true))
    mask_sum = np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred))
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def binarize_array(im):
    """Binarize image

    :param np.array im: Prediction or target array
    :return np.array im_bin: Flattened and binarized array
    """
    im_bin = (im.flatten() / im.max()) > .5
    return im_bin.astype(np.uint8)