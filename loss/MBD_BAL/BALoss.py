import torch
import numpy as np
import MBD
from loss.bce_loss import cross_entropy_loss2d


def boundary_awareness_loss(inputs, seed, label):
    EPM = inputs
    seed = seed.squeeze()
    label = label.squeeze()

    EPM = torch.sigmoid(EPM)
    EPM_s_clone = torch.clone(EPM)
    EPM_s_clone = (EPM_s_clone*255).squeeze().detach().cpu().numpy().astype(np.uint8)

    seed = np.array(seed.squeeze().detach().cpu().numpy()).astype(np.int32)
    print(np.unique(seed))
    saddle = MBD.geodesic_saddle(EPM_s_clone,seed)
    saddle = (saddle/255).astype(np.uint8)
    saddle = torch.from_numpy(np.array([saddle])).cuda()
    
    EPM_contour = (EPM * saddle)
    label_s = (label * saddle).unsqueeze(axis=0)
    BCE_loss = cross_entropy_loss2d(EPM_contour, label_s, True, 1.1)
    return BCE_loss
