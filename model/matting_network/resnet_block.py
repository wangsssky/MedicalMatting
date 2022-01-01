import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import init_weights


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), (stride, stride), (1, 1), bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, (3, 3), (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, (1, 1), (stride, stride)),
        nn.BatchNorm2d(out_channel))
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))

    layers = nn.Sequential(*layers)
    layers.apply(init_weights)
    return layers
