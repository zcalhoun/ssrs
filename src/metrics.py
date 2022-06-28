"""
This file implements various metrics that will be used
during training.

A lot of the code is taken from
https://github.com/bohaohuang/mrs/blob/master/mrs_utils/metric_utils.py
"""

# Pytorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def load(metric_name, device):
    if metric_name == "softiou":
        return SoftIoULoss(device)
    elif metric_name == "xent":
        return nn.BCEWithLogitsLoss()
    else:
        return NotImplementedError("This loss isn't set up yet.")


class LossClass(nn.Module):
    """
    The base class of loss metrics, all loss metrics should inherit from this class
    This class contains a function that defines how loss is computed (def forward) and a loss tracker that keeps
    updating the loss within an epoch
    """

    def __init__(self):
        super(LossClass, self).__init__()
        self.loss = 0
        self.cnt = 0

    def forward(self, pred, lbl):
        raise NotImplementedError

    def update(self, loss, size):
        """
        Update the current loss tracker
        :param loss: the computed loss
        :param size: #elements in the batch
        :return:
        """
        self.loss += loss.item() * size
        self.cnt += 1

    def reset(self):
        """
        Reset the loss tracker
        :return:
        """
        self.loss = 0
        self.cnt = 0

    def get_loss(self):
        """
        Get mean loss within this epoch
        :return:
        """
        return self.loss / self.cnt


class SoftIoULoss(LossClass):
    """
    Soft IoU loss that is differentiable
    This code comes from https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
    Paper: http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
    """

    def __init__(self, device, delta=1e-12):
        super(SoftIoULoss, self).__init__()
        self.name = "softIoU"
        self.device = device
        self.delta = delta

    def forward(self, pred, lbl):
        num_classes = pred.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[lbl.squeeze(1)].to(self.device)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(pred)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[lbl.squeeze(1)].to(self.device)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(pred, dim=1)
        true_1_hot = true_1_hot.type(pred.type())
        dims = (0,) + tuple(range(2, lbl.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2.0 * intersection / (cardinality + self.delta)).mean()
        return 1 - dice_loss


class CrossEntropyLoss(LossClass):
    """
    Cross entropy loss function used in training
    """

    def __init__(self, class_weights=(1.0, 1.0)):
        super(CrossEntropyLoss, self).__init__()
        self.name = "xent"
        # class_weights = torch.tensor([float(a) for a in class_weights])
        self.criterion = nn.CrossEntropyLoss(class_weights)

    def forward(self, pred, lbl):
        if len(lbl.shape) == 4 and lbl.shape[1] == 1:
            lbl = lbl[:, 0, :, :]
        return self.criterion(pred, lbl)
