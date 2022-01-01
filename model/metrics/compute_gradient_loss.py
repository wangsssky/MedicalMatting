"""
This code refers to the matlab code:
% compute the gradient error given a prediction, a ground truth and a trimap.
% author Ning Xu
% date 2018-1-1
"""
import numpy as np

from model.metrics.utils import mat2gray, gaussgradient


def compute_gradient_loss(pred, target, trimap=None):
	"""
	% pred: the predicted alpha matte
	% target: the ground truth alpha matte
	% trimap: the given trimap
	% step = 0.1
	"""
	pred_ = mat2gray(pred)
	target_ = mat2gray(target)
	pred_x, pred_y = gaussgradient(pred_, 1.4)
	target_x, target_y = gaussgradient(target_, 1.4)
	pred_amp = np.sqrt(pred_x**2 + pred_y**2)
	target_amp = np.sqrt(target_x**2 + target_y**2)

	error_map = (pred_amp - target_amp)**2
	if trimap is not None:
		loss = np.sum(np.sum(error_map * (trimap == 128)))
	else:
		loss = np.sum(np.sum(error_map))
	return loss/1000.0
