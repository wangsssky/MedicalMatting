"""
This code refers to the matlab code:
% compute the SAD error given a prediction, a ground truth and a trimap.
% author Ning Xu
% date 2018-1-1
"""
import numpy as np


def compute_sad_loss(pred, target, trimap=None):
	"""
	% the loss is scaled by 1000 due to the large images used in our experiment.
	% Please check the result table in our paper to make sure the result is correct.
	"""
	error_map = np.abs(pred.astype('int32')-target.astype('int32'))/255.0
	if trimap is not None:
		loss = np.sum(np.sum(error_map*(trimap == 128)))
	else:
		loss = np.sum(np.sum(error_map))
	loss = loss / 1000
	return loss
