import torch
import numpy as np

from model.metrics.utils import compute_dice_accuracy


def dice_at_thresh(labels, preds, thresh=0.5):
    pred_masks = []
    for pred in preds:
        pred_masks.append((pred > thresh).float())

    if len(preds) == 0:
        return 0.0

    dice_scores = []
    for mask in pred_masks:
        dice_ = []
        for label in labels:
            # dim0 is batch
            dice_instance = compute_dice_accuracy(label, mask)
            dice_.append(dice_instance.item())
        dice_scores.append(max(dice_))
    return dice_scores
