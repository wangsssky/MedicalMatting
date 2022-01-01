import torch
import torchvision.transforms as transforms

import cv2
import numpy as np


def tensor2im(image):
    shape = image.shape

    to_pil_image = transforms.ToPILImage()
    temp = to_pil_image(image.squeeze(0).cpu())
    img = np.asarray(temp)
    if len(shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def concat_images(*images, row=1):
    item_num = len(images)
    col = np.ceil(item_num / row)
    shape = images[0].shape

    col, row = int(col), int(row)

    concat_image = np.zeros([shape[0] * row, shape[1] * col, 3]).astype('uint8')
    idx = 0
    for r in range(row):
        for c in range(col):
            if idx >= item_num:
                break
            image = images[idx]
            if len(image.shape) != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            concat_image[r * shape[0]:(r + 1) * shape[0], c * shape[1]:(c + 1) * shape[1], :] = image
            if idx > 0:
                cv2.putText(concat_image, str(idx), ((c + 1) * shape[1] - 20, (r +1) * shape[0] - 20),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            idx += 1
    return concat_image


def mixup_image(image, mask):
    zero_mask = np.zeros_like(image)
    image[:, :, 0][mask > 0] = mask[mask > 0]
    zero_mask[:, :, 0][mask > 0] = mask[mask > 0]
    return cv2.addWeighted(image, 0.4, zero_mask, 0.6, 0)


def rand_seed(SEED=1234):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

