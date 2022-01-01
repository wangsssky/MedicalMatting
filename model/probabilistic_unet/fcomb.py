import torch
import torch.nn as nn
import numpy as np

from model.utils import init_weights, init_weights_orthogonal_normal


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that
    combines the sample taken from the latent space, and output of the
    UNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(self, num_filters, latent_dim, num_output_channels,
                 num_classes, num_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels  # output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.num_convs_fcomb = num_convs_fcomb
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            # Decoder of N x a 1x1 convolution followed by a ReLU activation
            # function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0] + self.latent_dim,
                                    self.num_filters[0],
                                    kernel_size=(1, 1)))
            layers.append(nn.ReLU(inplace=True))

            for idx in range(num_convs_fcomb - 2):
                layers.append(nn.Conv2d(self.num_filters[0],
                                        self.num_filters[0],
                                        kernel_size=(1, 1)))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0],
                                        self.num_classes,
                                        kernel_size=(1, 1))

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        order_index = order_index.to(a.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z, latent_type='fcomb'):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_size x no_channels x H x W.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same
        as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample
            # taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            if latent_type == 'fcomb':
                latent = self.layers(feature_map)
                output = self.last_layer(latent)
            else:
                latent = feature_map
                output = self.last_layer(self.layers(feature_map))

            return output, latent
