import torch
from torch import nn

from loss.bce_loss import cross_entropy_loss2d_sigmoid


def iterative_loss(preds, vgg_true, y_true, args):
    mse_loss = nn.MSELoss(reduction='sum')
    K, N = len(preds), len(vgg_true)
    loss_pred, loss_vgg = 0., 0.

    for i in range(K):
        loss_pred += (i + 1) * cross_entropy_loss2d_sigmoid(preds[0][0][i], y_true)
        for j in range(N):
            loss_vgg += (i + 1) * mse_loss(torch.sigmoid(preds[0][1][j]), vgg_true[j])

    coeff = 0.5 * args.K * (args.K + 1)
    bce_loss =  loss_pred / coeff
    mse_loss = loss_vgg / (coeff * N)
    loss_total = bce_loss + args.mu * mse_loss
    return bce_loss, args.mu*mse_loss, loss_total
