import torch
import torch.nn as nn
from model.utils import init_weights


class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a
    non-linear activation function Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, padding, pool=True, batch_norm=False):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2,
                                       stride=2,
                                       padding=0,
                                       ceil_mode=True))

        for i in range(3):
            if 0 == i:
                in_channel = input_dim
            else:
                in_channel = output_dim
            layers.append(nn.Conv2d(in_channel, output_dim,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(int(padding), int(padding))))
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_features=output_dim))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce
    the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, padding, bilinear=True, batch_norm=False):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim,
                                                   kernel_size=(2, 2),
                                                   stride=(2, 2))
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, padding,
                                        pool=False, batch_norm=batch_norm)

    def forward(self, x, bridge):
        if self.bilinear:
            # This operation may produce nondeterministic gradients when given tensors on a CUDA device.
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)
        
        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out
