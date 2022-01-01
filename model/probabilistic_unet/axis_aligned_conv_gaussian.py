import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from model.probabilistic_unet.encoder import Encoder


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with
    axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block,
                 latent_dim, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters,
                               self.no_convs_per_block, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(in_channels=num_filters[-1],
                                    out_channels=2 * self.latent_dim,
                                    kernel_size=(1, 1), stride=(1, 1))
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight,
                                mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, inputs, mask=None):
        # If segmentation is not none, concatenate the mask to the
        # channel axis of the input
        if mask is not None:
            self.show_img = inputs
            self.show_seg = mask
            inputs = torch.cat((inputs, mask), dim=1)
            self.show_concat = inputs
            self.sum_input = torch.sum(inputs)

        encoding = self.encoder(inputs)
        self.show_enc = encoding

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for
        # mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise
        # it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist
