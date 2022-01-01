import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from model.metrics.utils import compute_dice_accuracy


def normalized_l1_loss(alpha, alpha_pred, mask):
    loss = 0
    eps = 1e-6
    for i in range(alpha.shape[0]):
        if mask[i, ...].sum() > 0:
            loss = loss + torch.sum(torch.abs(alpha[i, ...] * mask[i, ...] - alpha_pred[i, ...] * mask[i, ...])) / (
                    torch.sum(mask[i, ...]) + eps)
        else:
            loss = loss + torch.sum(mask[i, ...]) + eps
    loss = loss / alpha.shape[0]

    return loss


class AlphaLoss(_Loss):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, alpha, alpha_pred, mask):
        return normalized_l1_loss(alpha, alpha_pred, mask)


class AlphaGradientLoss(_Loss):
    def __init__(self):
        super(AlphaGradientLoss, self).__init__()

    def forward(self, alpha, alpha_pred, mask):
        fx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(alpha.device)
        fx = fx.view((1, 1, 3, 3))
        fy = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(alpha.device)
        fy = fy.view((1, 1, 3, 3))

        G_x = F.conv2d(alpha, fx, padding=1)
        G_y = F.conv2d(alpha, fy, padding=1)
        G_x_pred = F.conv2d(alpha_pred, fx, padding=1)
        G_y_pred = F.conv2d(alpha_pred, fy, padding=1)

        loss = normalized_l1_loss(G_x, G_x_pred, mask) + normalized_l1_loss(G_y, G_y_pred, mask)

        return loss


# Dice loss
def dice_loss(label, mask):
    mask = (torch.sigmoid(mask) > 0.5).float()
    return 1.0 - compute_dice_accuracy(label, mask)
