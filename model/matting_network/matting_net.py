import torch
import torch.nn as nn

from model.utils import init_weights
from model.matting_network.cbam import CBAM
from model.matting_network.resnet_block import make_layer


class Matting_Net(nn.Module):
    def __init__(self, input_channels, num_latent_features, num_filters,
                 num_class, use_uncertainty_map=True, use_CBAM=True):
        super(Matting_Net, self).__init__()

        self.use_uncertainty_map = use_uncertainty_map
        self.use_CBAM = use_CBAM

        scale_ratio = 4
        block_num = 3

        in_channel_num = input_channels + num_latent_features
        if self.use_uncertainty_map:
            in_channel_num += 1
        self.block1 = make_layer(in_channel=in_channel_num,
                                 out_channel=num_filters,
                                 block_num=block_num,
                                 stride=1)

        if self.use_CBAM:
            self.cbam = CBAM(num_filters, num_filters, no_spatial=True)

        if self.use_uncertainty_map:
            self.block2 = make_layer(in_channel=num_filters + 1,
                                     out_channel=num_filters * scale_ratio,
                                     block_num=block_num,
                                     stride=1)
            self.block3 = make_layer(in_channel=num_filters * scale_ratio + 1,
                                     out_channel=num_filters * scale_ratio,
                                     block_num=block_num,
                                     stride=1)
        else:
            self.block2 = make_layer(in_channel=num_filters,
                                     out_channel=num_filters * scale_ratio,
                                     block_num=block_num,
                                     stride=1)
            self.block3 = make_layer(in_channel=num_filters * scale_ratio,
                                     out_channel=num_filters * scale_ratio,
                                     block_num=block_num,
                                     stride=1)

        self.out_layer = nn.Sequential(
            nn.Conv2d(num_filters * scale_ratio, num_filters, (3, 3), (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_class, (3, 3), (1, 1), (1, 1), bias=True)
        )
        self.out_layer.apply(init_weights)

    def forward(self, x, image, probs):
        prob = torch.mean(torch.cat(probs, dim=1), dim=1, keepdim=True)
        uncertain_map = - prob * torch.log2(prob + 1.0e-6) \
                        - (1.0 - prob) * torch.log2(1.0 - prob + 1.0e-6)

        if self.use_uncertainty_map:
            x = self.block1(torch.cat([x, image, uncertain_map], dim=1))
            x = self.cbam(x)
            x = self.block2(torch.cat([x, uncertain_map], dim=1))
            x = self.block3(torch.cat([x, uncertain_map], dim=1))
        else:
            x = self.block1(torch.cat([x, image], dim=1))
            x = self.cbam(x)
            x = self.block3(self.block2(x))

        x = self.out_layer(x)

        return x, uncertain_map
