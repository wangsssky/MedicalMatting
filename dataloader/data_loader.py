import torch
from torch.utils.data.dataset import Dataset

import os
import copy
import numpy as np
import cv2
import pickle


# Dataset for Medical Matting
class AlphaDataset(Dataset):
    def __init__(self, dataset_location, input_size=128):
        self.images = []
        self.mask_labels = []
        self.alphas = []
        self.series_uid = []

        # read dataset
        max_bytes = 2 ** 31 - 1
        data = {}
        print("Loading file", dataset_location)
        bytes_in = bytearray(0)
        file_size = os.path.getsize(dataset_location)
        with open(dataset_location, 'rb') as f_in:
            for _ in range(0, file_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        data.update(new_data)

        # load dataset
        for key, value in data.items():
            # image 0-255, alpha 0-255, mask [0,1]
            self.images.append(pad_im(value['image'], input_size))
            masks = []
            for mask in value['masks']:
                masks.append(pad_im(mask, input_size))
            self.mask_labels.append(masks)
            if 'alpha' in value.keys():
                self.alphas.append(pad_im(value['alpha'], input_size))
            else:
                self.alphas.append(None)
            self.series_uid.append(value['series_uid'])

        # check
        assert (len(self.images) == len(self.mask_labels) == len(self.series_uid))
        for image in self.images:
            assert np.max(image) <= 255 and np.min(image) >= 0
        for alpha in self.alphas:
            assert np.max(alpha) <= 255 and np.min(alpha) >= 0
        for mask in self.mask_labels:
            assert np.max(mask) <= 1 and np.min(mask) >= 0

        # free
        del new_data
        del data

    def __getitem__(self, index):
        image = copy.deepcopy(self.images[index])
        mask_labels = copy.deepcopy(self.mask_labels[index])
        alpha = copy.deepcopy(self.alphas[index])
        series_uid = self.series_uid[index]

        return image, mask_labels, alpha, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)


def pad_im(image, size, value=0):
    shape = image.shape
    if len(shape) == 2:
        h, w = shape
    else:
        h, w, c = shape

    if h == w:
        if h == size:
            padded_im = image
        else:
            padded_im = cv2.resize(image, (size, size), cv2.INTER_CUBIC)
    else:
        if h > w:
            pad_1 = (h - w) // 2
            pad_2 = (h - w) - pad_1
            padded_im = cv2.copyMakeBorder(image, 0, 0, pad_1, pad_2, cv2.BORDER_CONSTANT, value=value)
        else:
            pad_1 = (w - h) // 2
            pad_2 = (w - h) - pad_1
            padded_im = cv2.copyMakeBorder(image, pad_1, pad_2, 0, 0, cv2.BORDER_CONSTANT, value=value)
    if padded_im.shape[0] != size:
        padded_im = cv2.resize(padded_im, (size, size), cv2.INTER_CUBIC)

    return padded_im
