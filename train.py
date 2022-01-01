import torch
from torch.optim.lr_scheduler import _LRScheduler

from tensorboardX import SummaryWriter

import os
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import shutil

from config import *
from dataloader.data_spliter import AlphaDatasetSpliter
from dataloader.transform import elastic_transform
from model.medical_matting import MedicalMatting, ModelWithLoss
from evaluate import evaluate
from utils.logger import Logger
from utils.utils import rand_seed
from dataloader.utils import data_preprocess


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path (*.yaml)", required=True)
    parser.add_argument("--save_path", type=str, help="save path", default='')
    args = parser.parse_args()
    opt = Config(config_path=args.config)

    rand_seed(opt.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log & model folder
    if args.save_path == '':
        opt.MODEL_DIR += '_{}_{}'.format(opt.DATASET, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        opt.MODEL_DIR = args.save_path

    if not os.path.exists(opt.MODEL_DIR):
        os.mkdir(opt.MODEL_DIR)

    logger = Logger(opt.MODEL_NAME, path=opt.MODEL_DIR)
    writer = SummaryWriter(opt.MODEL_DIR)

    if not os.path.exists(os.path.join(opt.MODEL_DIR, 'params.yaml')):
        shutil.copy(args.config, os.path.join(opt.MODEL_DIR, 'params.yaml'))

    # dataset
    data_spliter = AlphaDatasetSpliter(opt=opt, input_size=opt.INPUT_SIZE)

    for fold_idx in range(opt.KFOLD):
        print('### {} of {} FOLD ###'.format(fold_idx + 1, opt.KFOLD))
        train_loader, test_loader = data_spliter.get_datasets(fold_idx=fold_idx)
        rand_seed(opt.RANDOM_SEED)

        # Training Config
        epochs = opt.EPOCH_NUM
        epoch_start = 0
        net = MedicalMatting(
            input_channels=opt.INPUT_CHANNEL, num_classes=1, num_filters=opt.NUM_FILTERS,
            latent_dim=opt.LATENT_DIM, num_convs_fcomb=4, batch_norm=opt.USE_BN,
            use_matting=opt.USE_MATTING, use_uncertainty_map=opt.UNCERTAINTY_MAP,
            num_sampling=opt.SAMPLING_NUM
        )
        net = ModelWithLoss(net, kl_scale=opt.KL_SCALE, reconstruction_scale=opt.RECONSTRUCTION_SCALE,
                            alpha_scale=opt.ALPHA_SCALE, alpha_gradient_scale=opt.ALPHA_GRADIENT_SCALE,
                            loss_strategy=opt.LOSS_STRATEGY)

        if opt.OPTIMIZER == 'ADAM':
            optimizer = torch.optim.Adam(
                net.parameters(), lr=opt.LEARNING_RATE, weight_decay=opt.WEIGHT_DECAY)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=opt.LEARNING_RATE, momentum=opt.MOMENTUM, weight_decay=opt.WEIGHT_DECAY,
                nesterov=True)

        warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * opt.WARM_LEN)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Resume
        if opt.RESUME_FROM > 0:
            ckpt = torch.load(
                os.path.join(opt.MODEL_DIR,
                             '{}_{}_{}_{}.pth'.format(opt.MODEL_NAME, opt.DATASET, fold_idx, opt.RESUME_FROM)))
            net.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt.keys():
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt.keys():
                scheduler.load_state_dict(ckpt['scheduler'])
            epoch_start = opt.RESUME_FROM

        net.to(device)

        # Training
        for epoch in range(epoch_start, epochs):
            net.train()
            print_str = '-------epoch {}/{}-------'.format(epoch+1, epochs)
            logger.write_and_print(print_str)

            for step, (patch, masks, alpha, _) in enumerate(tqdm(train_loader)):
                if alpha is None:
                    continue

                patch, mask, alpha = data_preprocess(
                    patch, masks, alpha, opt, elastic_transform=elastic_transform, training=True)
                patch = patch.to(device)
                mask = mask.to(device)
                alpha = alpha.to(device)

                # prepare data
                batches_done = len(train_loader) * epoch + step
                optimizer.zero_grad()

                loss, outputs = net.forward(
                    patch, mask, alpha, train_matting=epoch >= opt.TRAIN_MATTING_START_FROM,
                    epoch=epoch)
                if torch.isnan(loss):
                    logger.write_and_print('***** Warning: loss is NaN *****')
                    loss = torch.tensor(10000).to(device)

                # print loss
                print_str = '\n-----------loss-----------\n# total loss: {}'.format(loss.item())
                loss_names = ['kl', 'reconstruction', 'alpha', 'alpha_gradient']
                for name in loss_names:
                    if name not in outputs.keys():
                        continue
                    print_str += '\t' + name + ': {:.4f}'.format(outputs[name])
                print_str += '\n'
                if net.model.use_matting:
                    if net.loss_strategy == 'uncertain':
                        for var_id in range(net.task_num):
                            print_str += '\tlog_var{}: {:.4f}'.format(var_id, net.log_vars[var_id].item())
                if opt.PRT_LOSS:
                    logger.write_and_print(print_str)
                else:
                    logger.write(print_str)

                loss.backward()
                optimizer.step()

                writer.add_scalars('Loss', {'train fold_idx-{}'.format(fold_idx): loss.item()}, batches_done)

                if epoch <= opt.WARM_LEN:
                    warmup_scheduler.step()
                    # print('lr', optimizer.param_groups[0]['lr'])

            # log learning_rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalars('learning_rate', {'fold_idx-{}'.format(fold_idx): current_lr}, epoch)
            scheduler.step()

            # save model
            ckpt = {'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    }
            torch.save(ckpt, os.path.join(opt.MODEL_DIR,
                                          '{}_{}_{}_{}.pth'.format(opt.MODEL_NAME, opt.DATASET, fold_idx, epoch + 1)))

            # evaluate each epoch
            metrics_dict = evaluate(net.model, test_loader, device, opt)
            print_str = ''
            for key in metrics_dict.keys():
                print_str += key + ': {:.4f}  '.format(metrics_dict[key])
                writer.add_scalars(key, {'fold_idx-{}'.format(fold_idx): metrics_dict[key]}, epoch)
            logger.write_and_print(print_str)

        # evaluate each fold
        evaluate(net.model, test_loader, device, opt)

        ckpt = {'model': net.state_dict()}
        torch.save(ckpt, os.path.join(opt.MODEL_DIR,
                                      '{}_{}_{}_{}.pth'.format(opt.MODEL_NAME, opt.DATASET, fold_idx, epochs)))


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
