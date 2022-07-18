import torch
from torch import nn


def distance_softmax(inputs, targets):
    inputs = torch.softmax(inputs, 1).float()
    targets = targets.squeeze(0)

    CEloss = nn.CrossEntropyLoss(reduction='sum')
    if len(targets.shape) == 3:
        crossEntropy_loss_sum = CEloss(inputs, targets)
        return crossEntropy_loss_sum
    else:
        batch, _, _, _ = targets.shape
        for b in range(batch):
            if b == 0:
                crossEntropy_loss_sum = CEloss(torch.unsqueeze(inputs[b,...], 0), targets[b,...])
            else:
                crossEntropy_loss_sum += CEloss(torch.unsqueeze(inputs[b,...], 0), targets[b,...])
        return crossEntropy_loss_sum