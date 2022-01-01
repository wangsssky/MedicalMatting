import torch
import numpy as np
from dataloader.data_spliter import AlphaDatasetSpliter
from utils.utils import tensor2im

import cv2
import os
import argparse
from tqdm import tqdm

from config import *
from model.medical_matting import MedicalMatting, ModelWithLoss
from utils.utils import rand_seed
from dataloader.utils import generate_masks_by_alpha, data_preprocess
from model.metrics.generalised_energy_distance import generalized_energy_distance
from model.metrics.dice_accuracy import dice_at_thresh
from model.metrics.compute_sad_loss import compute_sad_loss
from model.metrics.compute_mse_loss import compute_mse_loss
from model.metrics.compute_gradient_loss import compute_gradient_loss
from model.metrics.compute_connectivity_error import compute_connectivity_error


def evaluate(net, val_loader, device, opt):
    GED_global, Dice05_global, SAD_global, MSE_global, CONN_global, GRAD_global \
        = 0, 0, 0, 0, 0, 0

    net.eval()
    with torch.no_grad():
        for val_step, (patch, masks, alpha, sid) in enumerate(tqdm(val_loader)):
            if alpha is None:
                continue

            patch, masks, alpha = data_preprocess(
                patch, masks, alpha, opt, elastic_transform=None, training=False)
            patch = patch.to(device)

            if opt.POSTERIOR_TARGET == 'alpha':
                masks = generate_masks_by_alpha(alpha, level_num=8, bottom=0.2, up=0.7)

            outputs = net.inference(patch, num_preds=net.num_sampling)

            predictions = []
            for prediction in outputs['predictions']:
                prediction = torch.squeeze(prediction, 0)
                predictions.append(prediction.detach().cpu())

            # Dice score
            dice_scores_iter = dice_at_thresh(masks, predictions, thresh=0.5)
            dice_score_iter = np.mean(dice_scores_iter)

            for i in range(len(predictions)):
                predictions[i] = predictions[i] > 0.5
            # GED score
            GED_iter, cross_iter, d0_iter, d1_iter = \
                generalized_energy_distance(masks, predictions)

            # calc loss
            GED_global += GED_iter
            Dice05_global += dice_score_iter

            if net.use_matting:
                pred_alpha = outputs['pred_alpha']
                pred_alpha = pred_alpha.detach().cpu()

                pred_alpha_uint8 = (pred_alpha * 255).squeeze().numpy().astype('uint8')
                alpha_uint8 = (alpha * 255).squeeze().numpy().astype('uint8')

                uncertainty_map = outputs['uncertainty_map']

                SAD_global += compute_sad_loss(pred_alpha_uint8, alpha_uint8)
                MSE_global += compute_mse_loss(pred_alpha_uint8, alpha_uint8)
                CONN_global = compute_connectivity_error(pred_alpha_uint8, alpha_uint8)
                GRAD_global += compute_gradient_loss(pred_alpha_uint8, alpha_uint8)

            if opt.VISUALIZE:
                vis_pred = tensor2im(predictions[0])
                concat_pred = np.zeros([vis_pred.shape[0], vis_pred.shape[1] * len(predictions)])
                concat_pred[:, :vis_pred.shape[1]] = vis_pred
                for idx in range(1, len(predictions)):
                    vis_pred = tensor2im(predictions[idx])
                    concat_pred[:, vis_pred.shape[1] * idx:vis_pred.shape[1] * (idx + 1)] = vis_pred

                cv2.imshow('predictions', concat_pred)

                if net.use_matting:
                    vis_alpha = tensor2im(pred_alpha)
                    cv2.imshow('pred alpha', vis_alpha)

                cv2.waitKey(0)

    # store in dict
    metrics_dict = {'GED': GED_global / len(val_loader),
                    'DICE_0.5': Dice05_global / len(val_loader)}

    if net.use_matting:
        metrics_dict['SAD'] = SAD_global / len(val_loader)
        metrics_dict['MSE'] = MSE_global / len(val_loader)
        metrics_dict['CONN'] = CONN_global / len(val_loader)
        metrics_dict['GRAD'] = GRAD_global / len(val_loader)

    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path (*.yaml)", required=True)
    parser.add_argument("--save_path", type=str, help="save path", required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = Config(config_path=args.config)
    rand_seed(opt.RANDOM_SEED)

    # dataset
    data_spliter = AlphaDatasetSpliter(opt=opt, input_size=opt.INPUT_SIZE)
    evaluate_records = []

    for fold_idx in range(opt.KFOLD):
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        print('#********{} of {} FOLD *******#'.format(fold_idx+1, opt.KFOLD))
        train_loader, test_loader = data_spliter.get_datasets(fold_idx=fold_idx)
        rand_seed(opt.RANDOM_SEED)

        net = MedicalMatting(
            input_channels=opt.INPUT_CHANNEL, num_classes=1, num_filters=opt.NUM_FILTERS,
            latent_dim=opt.LATENT_DIM, num_convs_fcomb=4, batch_norm=opt.USE_BN,
            use_matting=opt.USE_MATTING, use_uncertainty_map=opt.UNCERTAINTY_MAP,
            num_sampling=opt.SAMPLING_NUM)
        net = ModelWithLoss(net, kl_scale=opt.KL_SCALE, reconstruction_scale=opt.RECONSTRUCTION_SCALE,
                            alpha_scale=opt.ALPHA_SCALE, alpha_gradient_scale=opt.ALPHA_GRADIENT_SCALE,
                            loss_strategy=opt.LOSS_STRATEGY)

        ckpt = torch.load(os.path.join(args.save_path,
                                       '{}_{}_{}_{}.pth'.format(opt.MODEL_NAME, opt.DATASET,
                                                                fold_idx, opt.EPOCH_NUM)))
        net.load_state_dict(ckpt['model'])

        net.to(device)

        metrics_dict = evaluate(net.model, test_loader, device, opt)
        evaluate_records.append(metrics_dict)
        for key in metrics_dict.keys():
            print(key, ': ', metrics_dict[key])

    print(args.save_path)
    for key in evaluate_records[0].keys():
        temp = []
        for record in evaluate_records:
            temp.append(record[key])
        print('{}: {:.8f}±{:.8f}'.format(key, np.mean(temp), np.std(temp, ddof=0)))
