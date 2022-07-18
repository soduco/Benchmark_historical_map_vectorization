import torch
import torch.nn as nn
import numpy as np
from numpy import newaxis
from scipy.signal import convolve2d
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pdb
import MBD
import cv2
import gudhi as gd
import numpy
import math

def cross_entropy_loss2d(inputs, targets, cuda=True, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    inputs = torch.sigmoid(inputs)
    loss = nn.BCELoss(weights, reduction='sum')(inputs, targets)
    return loss

def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1,p2)

def geodesic_segment_loss(inputs, seed, label, iter_=3):
    # target_image: the boundary total
    EPM = inputs.squeeze()
    seed = seed.squeeze()
    label = label.squeeze()

    EPM = torch.sigmoid(EPM)
    EPM_s_clone = EPM.clone()
    for j in range(iter_):
        EPM_s_clone = soft_erode(EPM_s_clone)
    EPM_s_clone = (EPM_s_clone*255).detach().cpu().numpy().astype(np.uint8)

    seed = np.array(seed.detach().cpu().numpy()).astype(np.int32)

    saddle = MBD.geodesic_saddle(EPM_s_clone,seed)

    pdb.set_trace()
    # kernel = np.ones((3,3),np.uint8)
    # saddle = cv2.dilate(saddle,kernel,iterations = 1)

    # cv2.imwrite('check.png', (( EPM_s_clone/3 + saddle/3 + seed/3).astype(np.uint8) ))

    saddle = (saddle/255).astype(np.uint8)
    saddle = torch.from_numpy(np.array([saddle])).cuda()

    EPM_contour = (EPM * saddle).unsqueeze(axis=0)
    label_s = (label * saddle).unsqueeze(axis=0)

    # cv2.imwrite('abc.png', ((EPM_contour*255).squeeze().detach().cpu().numpy().astype(np.uint8) ))
    # cv2.imwrite('cba.png', ((label_s*255).squeeze().detach().cpu().numpy().astype(np.uint8) ))

    # BCE_loss = cross_entropy_loss2d(EPM_contour, label_s, True, 1.1)
    MSE_loss = nn.MSELoss(EPM_contour, label_s)

    return MSE_loss
