"""
This code refers to the matlab code:
% compute the MSE error given a prediction, a ground truth and a trimap.
% author Ning Xu
% date 2018-1-1
"""
import numpy as np


def compute_mse_loss(pred, target, trimap=None):
	"""
	% pred: the predicted alpha matte
	% target: the ground truth alpha matte
	% trimap: the given trimap
	"""
	error_map = np.array(pred.astype('int32')-target.astype('int32'))/255.0
	if trimap is not None:
		loss = sum(sum(error_map**2 * (trimap == 128))) / sum(sum(trimap == 128))
	else:
		h, w = pred.shape
		loss = sum(sum(error_map ** 2)) / (h*w)
	return loss
