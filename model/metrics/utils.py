import numpy as np
import cv2
import scipy.ndimage as ndimage
from skimage import measure


def mat2gray(m):
    mat = np.double(m)
    out = np.zeros(mat.shape, np.double)
    normalized = cv2.normalize(mat, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return normalized


def gauss(x, sigma):
    # Gaussian
    return np.exp(-x**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))


def dgauss(x, sigma):
    # first order derivative of Gaussian
    return -x * gauss(x, sigma) / sigma**2


def gaussgradient(IM, sigma):
    """
    %GAUSSGRADIENT Gradient using first order derivative of Gaussian.
    %  [gx,gy]=gaussgradient(IM,sigma) outputs the gradient image gx and gy of
    %  image IM using a 2-D Gaussian kernel. Sigma is the standard deviation of
    %  this kernel along both directions.
    %
    %  Contributed by Guanglei Xiong (xgl99@mails.tsinghua.edu.cn)
    %  at Tsinghua University, Beijing, China.
    """

    # determine the appropriate size of kernel.
    # The smaller epsilon, the larger size.
    epsilon = 1e-2
    halfsize = int(np.ceil(sigma*np.sqrt(-2*np.log(np.sqrt(2*np.pi)*sigma*epsilon))))
    size = 2 * halfsize + 1
    # generate a 2-D Gaussian kernel along x direction
    hx = np.zeros([size, size], dtype=float)
    for i in range(size):
        for j in range(size):
            u = [i-halfsize, j-halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx/np.sqrt(np.sum(np.sum(np.abs(hx)*np.abs(hx))))
    # generate a 2-D Gaussian kernel along y direction
    hy = np.transpose(hx)
    # 2-D filtering
    gx = cv2.filter2D(IM, -1, hx, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(IM, -1, hy, borderType=cv2.BORDER_REPLICATE)
    return gx, gy


def bwdist(IM):
    return ndimage.distance_transform_edt(1-IM)


def bwconncomp(IM, conn=1):
    label = measure.label(IM, connectivity=conn)
    return label


def compute_dice_accuracy(label, mask):
    smooth = 1.0
    batch = label.size(0)
    m1 = label.view(batch, -1).float()  # Flatten
    m2 = mask.view(batch, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

