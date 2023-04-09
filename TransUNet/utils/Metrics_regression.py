import numpy as np
import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision, Dice, F1Score, JaccardIndex, Precision, Recall, Specificity

from utils.config import ACCURACY, AUPRC, AUROC_, DICE_SCORE, F1_SCORE, JACCARD_INDEX, PRECISION, RECALL, \
    SPECIFICITY, PCC, R2, NRMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.metrics import normalized_root_mse
from scipy.stats import pearsonr


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


class Metrics():
    def __init__(self, device: torch.device, num_classes=2) -> None:
        # self.accuracy = Accuracy().to(device=device)
        # self.dice_score = Dice().to(device=device)
        # self.jaccard_index = JaccardIndex(num_classes=num_classes).to(device=device)
        # self.precision = Precision().to(device=device)
        # self.recall = Recall().to(device=device)
        # self.specificity = Specificity().to(device=device)
        # self.f1_score = F1Score().to(device=device)

        self.pcc = pearsonr
        self.r2 = r2_score
        self.nrmse = normalized_root_mse
        self.dice_score = mean_dice_np

        self.num_classes = num_classes
        self.metrics_history = {}
        self.metrics_history[ACCURACY] = []
        self.metrics_history[DICE_SCORE] = []
        self.metrics_history[JACCARD_INDEX] = []
        self.metrics_history[PRECISION] = []
        self.metrics_history[RECALL] = []
        self.metrics_history[SPECIFICITY] = []
        self.metrics_history[F1_SCORE] = []

    def reset(self) -> None:
        pass
        # self.accuracy.reset()
        # self.dice_score.reset()
        # self.jaccard_index.reset()
        # self.precision.reset()
        # self.recall.reset()
        # self.specificity.reset()
        # self.f1_score.reset()
        # self.auroc.reset()
        # self.auprc.reset()

    def update(self, input: torch.Tensor, target: torch.Tensor) -> dict:
        metric_dict = {}
        target = target.int()

        # metric_dict[ACCURACY] = self.accuracy(input, target)
        # metric_dict[DICE_SCORE] = self.dice_score(input, target)
        # metric_dict[JACCARD_INDEX] = self.jaccard_index(input, target)
        # metric_dict[PRECISION] = self.precision(input, target)
        # metric_dict[RECALL] = self.recall(input, target)
        # metric_dict[SPECIFICITY] = self.specificity(input, target)
        # metric_dict[F1_SCORE] = self.f1_score(input, target)
        metric_dict[PCC] = self.pcc(input.cpu().detach().numpy().flatten(), target.cpu().detach().numpy().flatten())[0]
        metric_dict[R2] = self.r2(input.cpu().numpy(), target.cpu().numpy())
        metric_dict[NRMSE] = self.nrmse(input.cpu().numpy(), target.cpu().numpy())
        metric_dict[DICE_SCORE] =self.dice_score(input.cpu().numpy(), target.cpu().numpy())

        return metric_dict

    def compute(self) -> dict:
        metric_dict = {}

        # metric_dict[ACCURACY] = self.accuracy.compute()
        metric_dict[DICE_SCORE] = self.dice_score.compute()
        metric_dict[JACCARD_INDEX] = self.jaccard_index.compute()
        # metric_dict[PRECISION] = self.precision.compute()
        # metric_dict[RECALL] = self.recall.compute()
        # metric_dict[SPECIFICITY] = self.specificity.compute()
        # metric_dict[F1_SCORE] = self.f1_score.compute()
        # metric_dict[AUROC_] = self.auroc.compute()
        # metric_dict[AUPRC] = self.auprc.compute()

        # self.metrics_history[ACCURACY].append(metric_dict[ACCURACY].cpu())
        self.metrics_history[DICE_SCORE].append(metric_dict[DICE_SCORE].cpu())
        self.metrics_history[JACCARD_INDEX].append(metric_dict[JACCARD_INDEX].cpu())
        # self.metrics_history[PRECISION].append(metric_dict[PRECISION].cpu())
        # self.metrics_history[RECALL].append(metric_dict[RECALL].cpu())
        # self.metrics_history[SPECIFICITY].append(metric_dict[SPECIFICITY].cpu())
        # self.metrics_history[F1_SCORE].append(metric_dict[F1_SCORE].cpu())
        # self.metrics_history[AUROC_].append(metric_dict[AUROC_].cpu())
        # self.metrics_history[AUPRC].append(metric_dict[AUPRC].cpu())

        return metric_dict

    def get_metrics(self) -> list:
        return self.metrics_history
