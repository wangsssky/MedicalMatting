import torch
import numpy as np
import random
import copy

from dataloader.transform import rotate, flip


def data_augmentation(patch, mask_labels, alpha, elastic_transform=None):
	if alpha is None:
		inputs = mask_labels + [patch]
	else:
		inputs = mask_labels + [alpha, patch]

	# flip
	inputs = flip(inputs)
	# rotate
	outputs = rotate(inputs)

	# elastic transform
	if elastic_transform is not None:
		# elastic transform
		image_num = len(outputs)
		im_stack = []
		for i in range(image_num):
			if len(inputs[i].shape) == 2:
				im_stack.append(inputs[i].astype('uint8')[..., None])
			else:
				im_stack.append(inputs[i].astype('uint8'))

		outputs = elastic_transform(im_stack)

		if alpha is None:
			masks, patch = \
				outputs[..., :len(mask_labels)], \
				np.squeeze(outputs[..., len(mask_labels):])
		else:
			masks, alpha, patch = \
				outputs[..., :len(mask_labels)], \
				outputs[..., len(mask_labels)], \
				np.squeeze(outputs[..., len(mask_labels) + 1:])

		assert len(mask_labels) == masks.shape[2]
		for i in range(len(mask_labels)):
			mask_labels[i] = masks[..., i]

	else:
		for i in range(len(mask_labels)):
			mask_labels[i] = outputs[i]
		if alpha is None:
			patch = outputs[len(mask_labels)]
		else:
			alpha, patch = outputs[len(mask_labels)], outputs[len(mask_labels)+1]
	return patch, mask_labels, alpha


def preprocess_func(patch, mask_labels, alpha, augmentation=True, elastic_transform=None):
	# augmentation
	if augmentation:
		patch, mask_labels, alpha = data_augmentation(
			patch, mask_labels, alpha, elastic_transform=elastic_transform)

	# expand dimension & convert to torch tensors
	if len(patch.shape) == 2:
		patch = np.expand_dims(patch, axis=0)
	else:
		# color patch
		patch = np.transpose(patch, (2, 0, 1))
	patch = torch.from_numpy(patch).type(torch.FloatTensor) / 255.0
	patch = patch.unsqueeze(0)

	if alpha is not None:
		alpha = torch.from_numpy(
			np.expand_dims(alpha, axis=0)).type(torch.FloatTensor) / 255.0
		alpha = alpha.unsqueeze(0)

	for i in range(len(mask_labels)):
		mask_labels[i] = torch.from_numpy(
			np.expand_dims(mask_labels[i], axis=0)).type(torch.FloatTensor)
		mask_labels[i] = (mask_labels[i] > 0.5).float().unsqueeze(0)
	return patch, mask_labels, alpha


def data_preprocess(patch, mask_labels, alpha, opt, elastic_transform=None, training=True):
	patch_list, mask_labels_list, alpha_list = [], [], []
	assert (patch.shape[0] == alpha.shape[0]) and (patch.shape[0] == mask_labels[0].shape[0])
	batch_num = patch.shape[0]
	for i in range(batch_num):
		patch_list.append(patch[i, ...].numpy().astype('uint8'))
		alpha_list.append(alpha[i, ...].numpy().astype('uint8'))
		m = []
		for l in range(len(mask_labels)):
			m.append(mask_labels[l][i, ...].numpy().astype('uint8'))
		mask_labels_list.append(m)

	patch_stack, mask_stack, alpha_stack = [], [], []

	if not training:
		assert batch_num == 1
		p, m_list, a = preprocess_func(
			patch_list[0], mask_labels_list[0], alpha_list[0],
			augmentation=False, elastic_transform=None)

		return p, m_list, a

	for i in range(batch_num):
		for t in range(opt.TRAIN_TIME_AUG):
			p, m_list, a = preprocess_func(
				copy.deepcopy(patch_list[i]),
				copy.deepcopy(mask_labels_list[i]),
				copy.deepcopy(alpha_list[i]), elastic_transform)
			patch_stack.append(p)
			# mask generate
			m = generate_posterior_target(
				method=opt.POSTERIOR_TARGET, masks=m_list,
				alpha=a, gen_type=opt.GEN_TYPE)
			mask_stack.append(m)
			alpha_stack.append(a)
	patch = torch.cat(patch_stack)
	mask = torch.cat(mask_stack)
	alpha = torch.cat(alpha_stack)

	return patch, mask, alpha


def generate_by_mask(masks, type='rand'):
	# random select one
	if type == 'rand':
		mask = masks[random.randint(0, len(masks)-1)]

	# random combine masks by union or intersection, or just ignore the mask.
	elif type == 'combine':
		random.shuffle(masks)
		mask = masks[0]

		for idx in range(1, len(masks)):
			if random.randint(0, 1) == 0:
				continue

			if random.randint(0, 1) == 0:
				mask = mask * masks[idx]
			else:
				mask = ((mask + masks[idx]) > 0).float()
	elif type == 'mean':
		mask = torch.mean(torch.cat(masks, dim=0), dim=0, keepdim=True)
	else:
		raise ValueError('Invalid type "{}".'.format(type))
	return mask


def generate_by_alpha(alpha, bottom=0.2, up=0.7, type='rand'):
	if type == 'rand':
		min_val = (alpha.min() * 255)
		interval = (alpha.max() - alpha.min()) * 255

		thresh = np.random.randint(
			int(min_val + interval * bottom),
			int(min_val + interval * up))
		thresh = thresh / 255.0
		mask = torch.ones_like(alpha)
		mask[alpha < thresh] = 0.0
	else:
		raise ValueError('Invalid type "{}".'.format(type))
	return mask


def generate_posterior_target(method, masks, alpha, gen_type):
	# generate by masks
	if method == 'mask':
		mask = generate_by_mask(masks=masks, type=gen_type)
	# generate by alpha matte
	elif method == 'alpha':
		if alpha is not None:
			mask = generate_by_alpha(alpha)
		else:
			raise ValueError('Error raised from generate_posterior_target: No alpha.')
	else:
		raise ValueError('UNKNOWN POSTERIOR_TARGET: {}'.format(method))
	return mask


def generate_masks_by_alpha(alpha, level_num, bottom=0.2, up=0.7):
	min_val = alpha.min()
	margin = alpha.max() - alpha.min()

	bottom = min_val + margin * bottom
	up = min_val + margin * up

	interval = up - bottom
	masks = []

	values = interval / float(level_num)
	for l in range(level_num):
		thresh = l * values + bottom

		mask = torch.ones_like(alpha)
		mask[alpha < thresh] = 0.0
		masks.append(mask)
	return masks
