import torch
import torch.nn as nn
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import matplotlib.pyplot as plt

import pdb


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
    return loss.type(torch.DoubleTensor)


def cross_entropy_loss2d_sigmoid(inputs, targets, cuda=True, balance=1.1):
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


def multi_direction_loss(inputs, targets):
    n, c, h, w = inputs.size()
    for i in range(c):
        x = inputs[:,i,:,:].unsqueeze(dim=0).double()
        y = targets[:,i,:,:].unsqueeze(dim=0).double()
        if i == 0:
            c_loss_p = cross_entropy_loss2d_sigmoid(x, y, balance=1)
            c_loss = torch.where(torch.isnan(c_loss_p), torch.zeros_like(c_loss_p), c_loss_p)
        else:
            c_loss_p = cross_entropy_loss2d_sigmoid(x, y, balance=1)
            c_loss_p = torch.where(torch.isnan(c_loss_p), torch.zeros_like(c_loss_p), c_loss_p)
            c_loss += c_loss_p
    c_loss /= c
    c_loss = c_loss.float().cuda()
    c_loss.require_grad = True
    return c_loss

def create_connectivity_gt(gt_image):
    gt_image = gt_image.squeeze().cpu().numpy()
    # Array padding
    location = [(2,2),(2,1),(2,0),
                (1,2),(1,0),
                (0,2),(0,1),(0,0)]

    # Convolution of eight different positions
    location_encode = np.zeros_like(gt_image)
    for index, l in enumerate(location):
        location_base_kernel = np.zeros((3, 3))
        location_base_kernel[l] = 1
        if index == 0:
            location_encode = convolve2d(gt_image, location_base_kernel, boundary='fill', mode='same')
        else:
            location_encode = np.dstack((location_encode, convolve2d(gt_image, location_base_kernel, boundary='fill', mode='same')))
    location_encode = np.transpose(location_encode, (2, 0, 1))
    location_encode = torch.from_numpy(location_encode).unsqueeze(axis=0).float().cuda()
    return location_encode.type(torch.int64)

def network_loss(preds, y_true):
    # Input is the eight connectivity predictions
    # Target is the eight connectivity ground truth
    connectivity_target = create_connectivity_gt(y_true)
    
    # Calculate loss for every direction
    connectivity_loss = multi_direction_loss(preds, connectivity_target)

    return connectivity_loss
