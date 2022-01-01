"""
This code refers to the matlab code:
% compute the connectivity error given a prediction, a ground truth and a trimap.
% author Ning Xu
% date 2018-1-1
"""
import numpy as np
from model.metrics.utils import bwdist, bwconncomp


def compute_connectivity_error(pred, target, trimap=None, step=0.1):
	"""
	% pred: the predicted alpha matte
	% target: the ground truth alpha matte
	% trimap: the given trimap
	% step = 0.1
	"""
	pred_ = pred/255.0
	target_ = target/255.0

	dimy, dimx = pred_.shape

	thresh_steps = [step * i for i in range(int(1.0/step) + 1)]
	l_map = -1 * np.ones_like(pred_)
	dist_maps = np.zeros([dimy, dimx, len(thresh_steps)])

	for ii in range(1, len(thresh_steps)):
		pred_alpha_thresh = pred_ >= thresh_steps[ii]
		target_alpha_thresh = target_ >= thresh_steps[ii]

		omega = np.zeros([dimy, dimx])
		cc = bwconncomp(pred_alpha_thresh * target_alpha_thresh, conn=1)
		size_vec = []
		max_label = np.max(np.max(cc))

		if max_label > 0:
			for id in range(1, max_label+1):
				size_vec.append(np.sum(np.sum(cc == id)))

			max_id = np.argmax(size_vec)
			omega[cc == (max_id+1)] = 1

		flag = (l_map == -1) * (omega == 0)
		l_map[flag == 1] = thresh_steps[ii-1]

		dist_maps[:, :, ii] = bwdist(omega)
		dist_maps[:, :, ii] = dist_maps[:, :, ii] / np.max(np.max(dist_maps[:, :, ii]))

	l_map[l_map == -1] = 1

	pred_d = pred_ - l_map
	target_d = target_ - l_map

	pred_phi = 1 - pred_d * (pred_d >= 0.15)

	target_phi = 1 - target_d * (target_d >= 0.15)

	if trimap is not None:
		loss = np.sum(np.sum(np.abs(pred_phi - target_phi) * (trimap == 128)))
	else:
		loss = np.sum(np.sum(np.abs(pred_phi - target_phi)))
	return loss/1000.0
