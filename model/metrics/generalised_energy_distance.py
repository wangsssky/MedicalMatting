import numpy as np
import torch


def iou(x, y, axis=-1):
	iou_ = (x & y).sum(axis) / (x | y).sum(axis)
	iou_[np.isnan(iou_)] = 1.
	return iou_


# exclude background
def distance(x, y):
	try:
		per_class_iou = iou(x[:, None], y[None, :], axis=-2)
	except MemoryError:
		per_class_iou = []
		for x_ in x:
			per_class_iou.append(iou(np.expand_dims(x_, axis=0), y[None, :], axis=-2))
		per_class_iou = np.concatenate(per_class_iou)
	return 1 - per_class_iou[..., 1:].mean(-1)


def calc_generalised_energy_distance(dist_0, dist_1, num_classes):
	dist_0 = dist_0.reshape((len(dist_0), -1))
	dist_1 = dist_1.reshape((len(dist_1), -1))
	dist_0 = dist_0.numpy().astype("int")
	dist_1 = dist_1.numpy().astype("int")

	eye = np.eye(num_classes)
	dist_0 = eye[dist_0].astype('bool')
	dist_1 = eye[dist_1].astype('bool')

	cross_distance = np.mean(distance(dist_0, dist_1))
	distance_0 = np.mean(distance(dist_0, dist_0))
	distance_1 = np.mean(distance(dist_1, dist_1))
	return cross_distance, distance_0, distance_1


# Metrics for Uncertainty
def generalized_energy_distance(labels, preds, thresh=0.5, num_classes=2):
	masks = []
	for pred in preds:
		masks.append((torch.sigmoid(pred) > thresh).float())

	labels = torch.cat(labels, 0)
	masks = torch.cat(masks, 0)

	cross, d_0, d_1 = calc_generalised_energy_distance(labels, masks, num_classes)
	GED = 2 * cross - d_0 - d_1

	return GED, cross, d_0, d_1
