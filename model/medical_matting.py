import torch
import torch.nn as nn

from model.loss_functions import AlphaLoss, AlphaGradientLoss
from model.probabilistic_unet.prob_unet import ProbabilisticUnet
from model.matting_network.matting_net import Matting_Net

from model.loss_strategy import gen_loss_weight


class MedicalMatting(nn.Module):
    def __init__(self,
                 input_channels=1,
                 num_classes=1,
                 num_filters=(32, 64, 128, 192),
                 latent_dim=6,
                 num_convs_fcomb=4,
                 batch_norm=True,
                 hidden_channel=32,
                 entropy_thresh=0.1,
                 num_sampling=16,
                 use_matting=True,
                 use_uncertainty_map=True):
        super(MedicalMatting, self).__init__()

        self.use_matting = use_matting
        self.entropy_thresh = entropy_thresh
        self.num_sampling = num_sampling

        self.prob_unet = ProbabilisticUnet(
            input_channels=input_channels, num_classes=num_classes,
            num_filters=num_filters, latent_dim=latent_dim,
            num_convs_fcomb=num_convs_fcomb, batch_norm=batch_norm)

        if self.use_matting:
            self.matting_net = Matting_Net(
                input_channels=input_channels, num_latent_features=num_filters[0],
                num_filters=hidden_channel, num_class=num_classes,
                use_uncertainty_map=use_uncertainty_map)

    def forward(self, patch, mask, train_matting=True):
        outputs = {}

        # prob.Unet
        self.prob_unet.forward(patch, mask)
        kl_loss, score_map, latent_features = self.prob_unet.elbo()

        outputs['kl'] = kl_loss
        outputs['score_map'] = score_map

        # prob.Unet + matting
        if self.use_matting and train_matting:
            predictions = self.prob_unet.posterior_sample(
                sample_num=self.num_sampling, reconstruct_posterior_mean=False)

            # from utils.utils import tensor2im
            # import cv2
            # vis_mask = tensor2im(uncertain_mask[0][0].cpu())
            # cv2.imshow('uncertain_mask', (vis_mask*255).astype('uint8'))
            # cv2.waitKey(0)

            pred_alpha, uncertainty_map = \
                self.matting_net.forward(latent_features, patch, predictions)

            outputs['pred_alpha'] = pred_alpha
            outputs['uncertainty_map'] = uncertainty_map
            outputs['predictions'] = predictions

        return outputs

    def inference(self, patch, num_preds=20):
        assert num_preds >= 1

        outputs = {}

        # prob.Unet
        self.prob_unet.forward(patch, mask=None)
        pred, latent_features = self.prob_unet.sample(testing=True)

        predictions = []
        if num_preds > 1:
            predictions = [torch.sigmoid(pred)]
            for i in range(num_preds-1):
                pred, _ = self.prob_unet.sample(testing=True)
                predictions.append(torch.sigmoid(pred))

        outputs['predictions'] = predictions

        # prob.Unet + Matting
        if self.use_matting:
            pred_alpha, uncertainty_map = \
                self.matting_net.forward(latent_features, patch, predictions)
            outputs['pred_alpha'] = torch.sigmoid(pred_alpha)
            outputs['uncertainty_map'] = uncertainty_map

        return outputs


class ModelWithLoss(nn.Module):
    def __init__(self, model, kl_scale=10, reconstruction_scale=1.0,
                 alpha_scale=1.0, alpha_gradient_scale=1.0, loss_strategy='OAWS'):
        super().__init__()

        self.model = model
        self.bcewithlogit = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction='none')
        self.kl_scale = kl_scale
        self.reconstruction_scale = reconstruction_scale
        self.loss_strategy = loss_strategy

        if self.model.use_matting:
            if self.loss_strategy == 'UWS':
                self.task_num = 2
                self.log_vars = nn.Parameter(torch.ones(self.task_num) * 4.0)
            elif self.loss_strategy == 'OAWS':
                pass
            elif self.loss_strategy == 'None':
                pass
            else:
                raise ValueError('Invalid loss strategy {}.'.format(self.loss_strategy))

            self.alpha_loss = AlphaLoss()
            self.alpha_gradient_loss = AlphaGradientLoss()
            self.alpha_scale = alpha_scale
            self.alpha_gradient_scale = alpha_gradient_scale

    def forward(self, patches, masks, alphas=None, train_matting=True, epoch=-1):
        outputs = self.model(patches, masks, train_matting=train_matting)

        # prob.Unet
        kl_loss = self.kl_scale * outputs['kl']
        outputs['kl'] = outputs['kl'].item()
        reconstruction_loss = torch.mean(self.bcewithlogit(input=outputs['score_map'], target=masks))
        outputs['reconstruction'] = reconstruction_loss.item()
        reconstruction_loss *= self.reconstruction_scale
        loss = kl_loss + reconstruction_loss
        outputs['score_map'] = torch.sigmoid(outputs['score_map'])

        if self.model.use_matting and train_matting:
            uncertainty_map = outputs['uncertainty_map']
            pred_alpha = outputs['pred_alpha']

            uncertain_mask = torch.ones_like(uncertainty_map)
            uncertain_mask[uncertainty_map < self.model.entropy_thresh] = 0

            pred_alpha = torch.sigmoid(pred_alpha)
            # mask = torch.ones_like(alpha, device=alpha.device)

            alpha_loss = self.alpha_loss(alphas, pred_alpha, torch.ones_like(uncertain_mask))
            alpha_loss = self.alpha_scale * alpha_loss
            alpha_grad_loss = self.alpha_gradient_loss(alphas, pred_alpha, uncertain_mask)
            alpha_grad_loss = self.alpha_gradient_scale * alpha_grad_loss

            outputs['alpha'] = alpha_loss.item()
            outputs['alpha_gradient'] = alpha_grad_loss.item()

            if self.loss_strategy == 'UWS':
                loss = torch.exp(-self.log_vars[0]) * loss \
                       + torch.exp(-self.log_vars[1]) * 0.5 * (alpha_loss + alpha_grad_loss) \
                       + self.log_vars.sum()
            elif self.loss_strategy == 'OAWS':
                weight = gen_loss_weight(epoch, a=0.05, b=0.03)
                loss = weight * loss + (1-weight) * (alpha_loss + alpha_grad_loss)
            elif self.loss_strategy == 'None':
                loss = loss + alpha_loss + alpha_grad_loss
            else:
                raise ValueError('Invalid loss strategy {}.'.format(self.loss_strategy))

        return loss, outputs



