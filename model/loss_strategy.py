import numpy as np


def gen_loss_weight(epoch, a, b):
	return 0.5 * np.exp(-a*epoch) * np.cos(b*epoch * epoch) + 0.5
