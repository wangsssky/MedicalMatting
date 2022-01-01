# This code is based on: https://github.com/SimonKohl/probabilistic_unet

from model.unet.unet_blocks import *
from model.unet.unet import Unet
from torch.distributions import kl


from model.probabilistic_unet.axis_aligned_conv_gaussian import AxisAlignedConvGaussian
from model.probabilistic_unet.fcomb import Fcomb


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    num_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=(32, 64, 128, 192),
                 latent_dim=6, num_convs_fcomb=4, batch_norm=False):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.num_convs_per_block = 3
        self.num_convs_fcomb = num_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters,
                         apply_last_layer=False, padding=True, batch_norm=batch_norm)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters,
                                             self.num_convs_per_block, self.latent_dim)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters,
                                                 self.num_convs_per_block, self.latent_dim,
                                                 posterior=True)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels,
                           self.num_classes, self.num_convs_fcomb,
                           {'w': 'orthogonal', 'b': 'normal'}, use_tile=True)

    def forward(self, patch, mask):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if self.training:
            self.posterior_latent_space = self.posterior.forward(patch, mask)

        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch, False)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if not testing:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here.
            # For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def posterior_sample(self, sample_num=20, reconstruct_posterior_mean=False):
        """
        Repeat sampling during training
        """
        samples = []

        for _ in range(sample_num):
            z_posterior = self.posterior_latent_space.sample()
            samples.append(torch.sigmoid(
                self.reconstruct(use_posterior_mean=reconstruct_posterior_mean,
                                 calculate_posterior=False, z_posterior=z_posterior)[0]))
        return samples

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample)
        and UNet feature map use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can
        sample here or supply a sample
        """
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        z_posterior = self.posterior_latent_space.rsample()
        kl = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        output, latent_features = \
            self.reconstruct(use_posterior_mean=reconstruct_posterior_mean,
                             calculate_posterior=False, z_posterior=z_posterior)
        return kl, output, latent_features
